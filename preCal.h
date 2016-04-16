#ifndef PRECAL_H
#define PRECAL_H

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <iostream>
#include "parameter.h"

#define BLOCK_W 16
#define BLOCK_H 16
#define BLOCK_SIZE1 BLOCK_W*BLOCK_H
#define _USE_MATH_DEFINES
#include <math.h>

using namespace cv;

__global__
void variance_kernel(float* variance, const gpu::PtrStepSz<float> imgptr, const int2 dim) {
  
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;
  const int tid = tidy*BLOCK_W + tidx;
  const int x = blockIdx.x*BLOCK_W + tidx;
  const int y = blockIdx.y*BLOCK_H + tidy;
  
  if (x < 0 || x >= dim.x || y < 0 || y >= dim.y)
    return;

  const int wW = BLOCK_W + 2;
  const int wH = BLOCK_H + 2;
  const int wSize = wW*wH;
  __shared__ float window[wSize];
  
  // move data to shared
  int wXstart = blockIdx.x*BLOCK_W - 1;
  int wYstart = blockIdx.y*BLOCK_H - 1;
  for (int i = tid; i < wSize; i += BLOCK_SIZE1) {
    int wX = (i % wW) + wXstart;
    int wY = (i / wH) + wYstart;
    if (wX < 0 || wX >= dim.x || wY < 0 || wY >= dim.y)
      window[i] = 2;
    else
      window[i] = imgptr(wY, wX);
  }
  __syncthreads();
  
  // find max
  float max = 0;
  float value = imgptr(y, x);
  for (int idy = tidy; idy < tidy + 3; idy++)
    for (int idx = tidx; idx < tidx + 3; idx++) {
      int id = idy*wW + idx;
      if (window[id] != 2) {
        float diff = abs(value - window[id]);
        if (diff > max)
          max = diff;
      }
    }
  variance[y*dim.x + x] = max;
}

double calSigmaValue(const gpu::GpuMat &marker_d, const parameter &para, const bool &verbose) {

  // convert to gray channel
  gpu::GpuMat marker_g(para.mDim.x, para.mDim.y, CV_32FC1);
  gpu::cvtColor(marker_d, marker_g, CV_BGR2GRAY);
  gpu::GpuMat tmp(para.mDim.x, para.mDim.y, CV_32FC1);
  const int area = para.mDim.x * para.mDim.y;
  thrust::device_vector<float> variance(area, 0.0);
  
  // kernel parameter for TV
  dim3 bDim(BLOCK_W, BLOCK_H);
  dim3 gDim(para.mDim.x/bDim.x + 1, para.mDim.y/bDim.y + 1);

  // start calculation
  double blur_sigma = 1;
  float TVperNN = FLT_MAX;
  const float threshold = 1870 * (para.Btz.x*para.Btz.y) / (2 * para.Sf.x*para.markerDim.x) / (2 * para.Sf.y*para.markerDim.y) * area;
  while (TVperNN > threshold) {
    blur_sigma++;
    int kSize = 4 * blur_sigma + 1;
    kSize = (kSize <= 32) ? kSize : 31;
    gpu::GaussianBlur(marker_g, tmp, Size(kSize, kSize), blur_sigma);
    variance_kernel<<<gDim,bDim>>>(thrust::raw_pointer_cast(variance.data()), tmp, para.mDim);
    TVperNN = thrust::reduce(variance.begin(), variance.end());
    if (verbose)
      cout << "  calSigma ing..." << blur_sigma << ", " << TVperNN << endl;
  }
  return blur_sigma;
}

void preCal(parameter &para, gpu::GpuMat &marker_d, gpu::GpuMat &img_d, const cv::Mat &marker, const cv::Mat &img, 
            const float2 &Sf, const int2 &P, const float &delta, const float2 &Btz, const bool &verbose) {
  // allocate memory to GPU
  gpu::GpuMat marker0(marker);
  gpu::GpuMat img0(img);
  gpu::GpuMat marker1(para.mDim.x, para.mDim.y, CV_32FC3);
  gpu::GpuMat img1(para.mDim.x, para.mDim.y, CV_32FC3);
  gpu::GpuMat marker2(para.mDim.x, para.mDim.y, CV_32FC3);
  gpu::GpuMat img2(para.mDim.x, para.mDim.y, CV_32FC3);
  gpu::GpuMat img3(para.mDim.x, para.mDim.y, CV_32FC3);
  marker0.convertTo(marker1, CV_32FC3, 1.0/255.0);
  img0.convertTo(img1, CV_32FC3, 1.0 / 255.0);
  
  // dim of images
  para.mDim.x = marker.cols;
  para.mDim.y = marker.rows;
  para.iDim.x = img.cols;
  para.iDim.y = img.rows;
  
  // intrinsic parameter
  para.Sf = Sf;
  para.P = P;
  
  // search range in pose domain
  float wm = float(marker.cols);
  float hm = float(marker.rows);
  float minmDim = fmin(hm, wm);
  para.markerDim.x = wm / minmDim * 0.5;
  para.markerDim.y = hm / minmDim * 0.5;
  float minmarkerDim = fmin(para.markerDim.x, para.markerDim.y);
  para.Btz = Btz;
  para.Brx = make_float2(0, 80 * M_PI / 180);
  para.Brz = make_float2(-M_PI, M_PI);
  
  //bounds
  float m_tz = sqrt(Btz.x*Btz.y);
  float sqrt2 = sqrt(2);
  para.delta = delta;
  para.s4 = make_float4(delta/sqrt2/m_tz, delta/sqrt2/m_tz, delta/sqrt2/m_tz, delta/sqrt2/m_tz);
  para.s2 = make_float2(delta*sqrt2/m_tz, delta*sqrt2/m_tz);
  
  // smooth images
  double blur_sigma = calSigmaValue(marker1, para, verbose);
  if (verbose)
    std::cout << "blur sigma : " << blur_sigma << std::endl;
  int kSize = 4 * blur_sigma + 1;
  kSize = (kSize <= 32) ? kSize : 31;
  Ptr<gpu::FilterEngine_GPU> filter = gpu::createGaussianFilter_GPU(CV_32FC3, Size(kSize, kSize), blur_sigma);
  filter->apply(marker1, marker2, cv::Rect(0, 0, para.mDim.x, para.mDim.y));
  filter->apply(img1, img2, cv::Rect(0, 0, para.iDim.x, para.iDim.y));
  filter.release();
  
  // rgb2ycbcr
  gpu::cvtColor(marker2, marker_d, CV_BGR2YCrCb);
  gpu::cvtColor(img2, img3, CV_BGR2YCrCb);
  gpu::cvtColor(img3, img_d, CV_BGR2BGRA, 4);
  
  // bind img to texture mem
  tex_imgYCrCb.addressMode[0] = cudaAddressModeBorder;
  tex_imgYCrCb.addressMode[1] = cudaAddressModeBorder;
  tex_imgYCrCb.filterMode = cudaFilterModePoint;
  tex_imgYCrCb.normalized = false;
  cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
  cudaBindTexture2D(0, tex_imgYCrCb, img_d.data, desc, para.iDim.x, para.iDim.y, img_d.step);
}

#endif