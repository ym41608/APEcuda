#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <iostream>
#include <chrono>
#include "parameter.h"

#define BLOCK_W 8
#define BLOCK_H 8
#define BLOCK_SIZE BLOCK_W*BLOCK_H
#define _USE_MATH_DEFINES
#include <math.h>

using namespace cv;

// Timer
class Timer {
  typedef std::chrono::time_point<std::chrono::high_resolution_clock> Clock;
  long long count;
  bool running;
  Clock prev_start_;
  Clock Now() {
    return std::chrono::high_resolution_clock::now();
  }
public:
  void Start() {
    running = true;
    prev_start_ = Now();
  }
  void Pause() {
    if (running) {
      running = false;
      auto diff = Now() - prev_start_;
      count += std::chrono::duration_cast<std::chrono::nanoseconds>(diff).count();
    }
  }
  void Reset() {
    running = false;
    count = 0;
  }
  long long get_count() {
    return count;
  }
  Timer() { Reset(); }
};

__global__
void variance_kernel(float* variance, const cv::cuda::PtrStepSz<float> imgptr, const int2 dim) {
  
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;
  const int tid = tidy*BLOCK_W + tidx;
  const int x = blockIdx.x*BLOCK_W + tidx;
  const int y = blockIdx.y*BLOCK_H + tidy;
  
  if (x < 0 || x >= dim.x || y < 0 || y > dim.y)
    return;

  const int wW = BLOCK_W + 2;
  const int wH = BLOCK_H + 2;
  const int wSize = wW*wH;
  __shared__ float window[wSize];
  
  // move data to shared
  int wXstart = blockIdx.x*BLOCK_W - 1;
  int wYstart = blockIdx.y*BLOCK_H - 1;
  for (int i = tid; i < wSize; i += BLOCK_SIZE) {
    int wX = (i % wW) + wXstart;
    int wY = (i / wH) + wYstart;
    if (wX < 0 || wX >= dim.x || wY < 0 || wY > dim.y)
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

double calSigmaValue(const cv::cuda::GpuMat &marker_d, const parameter &para) {

  // convert to gray channel
  cuda::GpuMat marker_g;
  cuda::cvtColor(marker_d, marker_g, CV_BGR2GRAY);
  cuda::GpuMat tmp;
  const int area = para.mDim.x * para.mDim.y;
  thrust::device_vector<float> variance(area);
  
  // kernel parameter for TV
  dim3 bDim(BLOCK_W, BLOCK_H);
  dim3 gDim(para.mDim.x/bDim.x + 1, para.mDim.y/bDim.y + 1);
  
  // start calculation
  double blur_sigma = 1;
  float TVperNN = FLT_MAX;
  const float threshold = 1870 * (para.Btz.x*para.Btz.y) / (2 * para.Sf.x*para.markerDim.x) / (2 * para.Sf.y*para.markerDim.y) * area;
  while (TVperNN > threshold) {
    blur_sigma++;
    Ptr<cuda::Filter> filter = cuda::createGaussianFilter(CV_32F, CV_32F, Size(blur_sigma * 4 + 1, blur_sigma * 4 + 1), blur_sigma);
    filter->apply(marker_g, tmp);
    variance_kernel<<<gDim,bDim>>>(thrust::raw_pointer_cast(variance.data()), tmp, para.mDim);
    TVperNN = thrust::reduce(variance.begin(), variance.end());
  }
  return blur_sigma;
}

void preCal(parameter &para, cv::cuda::GpuMat &marker_d, cv::cuda::GpuMat &img_d, const cv::Mat &marker, const cv::Mat &img, 
            const float2 &Sf, const int2 &P, const float &delta, const float &minTz, const float &maxTz, const bool &verbose) {
  // move to GPU
  cuda::GpuMat marker0(marker);
  cuda::GpuMat img0(img);
  cuda::GpuMat marker1;
  cuda::GpuMat img1;
  marker0.convertTo(marker1, CV_32FC3, 1.0/255.0);
  img0.convertTo(img1, CV_32FC3, 1.0 / 255.0);
  
  // dim of images
  para.mDim.x = marker.cols;
  para.mDim.y = marker.rows;
  para.iDim.x = img.cols;
  para.iDim.y = img.rows;
  
  // intrinsic parameter
  para.Sf.x = Sf.x;
  para.Sf.y = -Sf.y;
  para.P = P;
  
  // search range in pose domain
  float wm = float(marker.cols);
  float hm = float(marker.rows);
  float wi = float(img.cols);
  float hi = float(img.rows);
  float minmDim = fmin(hm, wm);
  para.markerDim.x = wm / minmDim * 0.5;
  para.markerDim.y = hm / minmDim * 0.5;
  float minmarkerDim = fmin(para.markerDim.x, para.markerDim.y);
  float tx_w = (para.P.x*maxTz)/para.Sf.x - minmarkerDim;
  float ty_w = (para.P.y*maxTz)/para.Sf.y - minmarkerDim;
  para.Btx = make_float2(-tx_w, tx_w);
  para.Bty = make_float2(-ty_w, ty_w);
  para.Btz = make_float2(minTz, maxTz);
  para.Brx = make_float2(0, 80 * M_PI / 180);
  para.Brz = make_float2(-M_PI, M_PI);
  
  //bounds
  float m_tz = sqrt(minTz*maxTz);
  float sqrt2 = sqrt(2);
  para.s4 = make_float4(delta/sqrt2/m_tz, delta/sqrt2/m_tz, delta/sqrt2/m_tz, delta/sqrt2/m_tz);
  para.s2 = make_float2(delta*sqrt2/m_tz, delta*sqrt2/m_tz);
  
  // smooth images
  cuda::GpuMat marker2;
  cuda::GpuMat img2;
  double blur_sigma = calSigmaValue(marker1, para);
  if (verbose)
    std::cout << "blur sigma : " << blur_sigma << std::endl;
  Ptr<cuda::Filter> filter = cuda::createGaussianFilter(CV_32FC3, CV_32FC3, Size(blur_sigma * 4 + 1, blur_sigma * 4 + 1), blur_sigma);
  filter->apply(marker1, marker2);
  filter->apply(img1, img2);

  // output
  //Mat mtmp, itmp;
  //marker2.download(mtmp);
  //img2.download(itmp);
  //mtmp.convertTo(mtmp, CV_8UC3, 255);
  //itmp.convertTo(itmp, CV_8UC3, 255);
  //imwrite("marker.png", mtmp);
  //imwrite("img.png", itmp);
  
  // rgb2ycbcr
  cuda::cvtColor(marker2, marker_d, CV_BGR2YCrCb);
  cuda::cvtColor(img2, img_d, CV_BGR2YCrCb);
}

int main() {
  Mat marker = cv::imread("Philadelphia.png");
  Mat img = cv::imread("10.png");

  cuda::GpuMat marker_d, img_d;
  parameter para;
  Timer timer;
  timer.Reset(); timer.Start();
  preCal(para, marker_d, img_d, marker, img, make_float2(1000, -1000), make_int2(400, 300), 0.25, 3, 8, true);
  timer.Pause();
  std::cout << "Time: " << timer.get_count() << " ns." << std::endl;
}