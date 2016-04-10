#ifndef RANPIXELS_H
#define RANPIXELS_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>

// constant
const static int SAMPLE_NUM = 444;
__constant__ float2 const_Mcoor[SAMPLE_NUM];
__constant__ float4 const_marker[SAMPLE_NUM];


struct intwhprg {
  int w, h;
  
  __host__ __device__
  intwhprg(int _w = 0, int _h = 100) {
    w = _w;
    h = _h;
  };
  __host__ __device__
  int2 operator()(const int n) const {
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> distw(-1, w - 1);
    thrust::uniform_int_distribution<int> disth(-1, h - 1);
    rng.discard(n);
    return make_int2(distw(rng), disth(rng));
  };
};

__global__ void assign_kernel(float2 *mCoor, float4 *mValue, int2 *rand_coor, const cv::cuda::PtrStepSz<float3> marker_dptr, const int2 mDim, const float2 markerDim) {
  const int tIdx = threadIdx.x;
  const int Idx = blockIdx.x * 128 + tIdx;
  if (Idx >= SAMPLE_NUM)
    return;
  
  int x = rand_coor[Idx].x;
  int y = rand_coor[Idx].y;
  
  float3 v = marker_dptr(y, x);
  mValue[Idx] = make_float4(v.x, v.y, v.z, 0);
  
  float2 coor;
  coor.x = (2 * float(x) - mDim.x) / mDim.x * markerDim.x;
  coor.y = -(2 * float(y) - mDim.y) / mDim.y * markerDim.y;
  mCoor[Idx] = coor;
};

void randSample(thrust::device_vector<float2>* mCoor, thrust::device_vector<float4>* mValue, const cv::cuda::GpuMat &marker_d, const int2& mDim, const float2 markerDim) {

  // rand pixel
  thrust::device_vector<int2> rand_coor(SAMPLE_NUM, make_int2(0, 0));
  thrust::counting_iterator<int> i0(58);
  thrust::transform(i0, i0 + SAMPLE_NUM, rand_coor.begin(), intwhprg(mDim.x, mDim.y));

  // get pixel value and position
  const int BLOCK_NUM = (SAMPLE_NUM - 1) / 128 + 1;
  assign_kernel << < BLOCK_NUM, 128 >> > (thrust::raw_pointer_cast(mCoor->data()), thrust::raw_pointer_cast(mValue->data()), thrust::raw_pointer_cast(rand_coor.data()), marker_d, mDim, markerDim);

  // bind to const mem
  cudaMemcpyToSymbol(const_Mcoor, thrust::raw_pointer_cast(mCoor->data()), sizeof(float2)* SAMPLE_NUM, 0, cudaMemcpyDeviceToDevice);
  cudaMemcpyToSymbol(const_marker, thrust::raw_pointer_cast(mValue->data()), sizeof(float4)* SAMPLE_NUM, 0, cudaMemcpyDeviceToDevice);
};

#endif