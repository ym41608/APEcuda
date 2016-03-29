#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>

using namespace cv;
using namespace std;


// Texture
texture<float4, cudaTextureType2D, cudaReadModeElementType> tex_imgYCrCb;

// Constant
const static int SAMPLE_NUM = 444;
__constant__ float2 const_Mcoor[SAMPLE_NUM];
__constant__ float4 const_marker[SAMPLE_NUM];

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

struct intwhprg {
  int w, h;
  
  __host__ __device__
  intwhprg(int _w = 0, int _h = 100) {
    w = _w;
    h = _h;
  }
  __host__ __device__
  int2 operator()(const int n) const {
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> distw(-1, w - 1);
    thrust::uniform_int_distribution<int> disth(-1, h - 1);
    rng.discard(n);
    return make_int2(distw(rng), disth(rng));
  }
};

__global__
void assign_kernel(float2 *mCoor, float4 *mValue, int2 *rand_coor, const cuda::PtrStepSz<float3> marker_d, const int2 mDim, const float2 markerDim) {
  const int tIdx = threadIdx.x;
  const int Idx = blockIdx.x * 128 + tIdx;
  if (Idx >= SAMPLE_NUM)
    return;
  
  int x = rand_coor[Idx].x;
  int y = rand_coor[Idx].y;
  
  float3 v = marker_d(y, x);
  mValue[Idx] = make_float4(v.x, v.y, v.z, 0);
  
  float2 coor;
  coor.x = (2 * float(x) - mDim.x) / mDim.x * markerDim.x;
  coor.y = -(2 * float(y) - mDim.y) / mDim.y * markerDim.y;
  mCoor[Idx] = coor;
}

void randSample(thrust::device_vector<float2>* mCoor, thrust::device_vector<float4>* mValue, const cuda::GpuMat &marker_d, const int2& mDim, const float2 markerDim) {

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
}

int main() {
  cout << cudaGetErrorString(cudaGetLastError()) << endl;
  // read image
  Mat marker = imread("marker.png");
  marker.convertTo(marker, CV_32FC3, 1 / 255.0);
  cvtColor(marker, marker, CV_BGR2YCrCb);
  cuda::GpuMat marker_d(marker);
  
  // allocate mem
  thrust::device_vector<float2> mCoor(SAMPLE_NUM, make_float2(0, 0));
  thrust::device_vector<float4> mValue(SAMPLE_NUM, make_float4(0, 0, 0, 0));
  //float2* mCoor;
  //float4* mValue;
  //cudaMalloc((void**)&mCoor, sizeof(float2) * SAMPLE_NUM);
  //cudaMalloc((void**)&mValue, sizeof(float4) * SAMPLE_NUM);
  
  // initial parmeter
  int2 mDim = make_int2(marker.cols, marker.rows);
  float2 markerDim = make_float2(0.5 * marker.cols / marker.rows, 0.5);
  
  // rand pixel
  //thrust::device_vector<int2> rand_coor(SAMPLE_NUM, make_int2(0, 0));
  //thrust::counting_iterator<int> i0(58);
  //thrust::transform(i0, i0 + SAMPLE_NUM, rand_coor.begin(), intwhprg(mDim.x, mDim.y));

  // rand sample
  Timer timer;
  timer.Reset(); timer.Start();
  randSample(&mCoor, &mValue, marker_d, mDim, markerDim);
  cudaDeviceSynchronize();
  timer.Pause();
  cout << "GPU: " << timer.get_count() << " ns." << endl;
  //ofstream outFile("outCuda.txt");
  //if (!outFile)
  //  return 0;
  //for (int i = 0; i < SAMPLE_NUM; i++) {
  //  float4 f4 = mValue[i];
  //  float2 f2 = mCoor[i];
  //  outFile << f4.x << " " << f4.y << " " << f4.z << " " << f4.w << " ";
  //  outFile << f2.x << " " << f2.y << endl;
  //}
  //outFile.close();
  //cudaFree((void*) mCoor);
  //cudaFree((void*) mValue);

  // initial constant memory
  float2 *coor2 = new float2[SAMPLE_NUM];
  float4 *value = new float4[SAMPLE_NUM];
  timer.Reset(); timer.Start();
  for (int i = 0; i < SAMPLE_NUM; i++) {
    //int2 xy = rand_coor[i];
    int x = rand() % mDim.x;
    int y = rand() % mDim.y;
    Vec3f YCrCb = marker.at<Vec3f>(y, x);
    value[i] = make_float4(YCrCb[0], YCrCb[1], YCrCb[2], 0);
    coor2[i].x = (2 * float(x) - mDim.x) / mDim.x * markerDim.x;
    coor2[i].y = -(2 * float(y) - mDim.y) / mDim.y * markerDim.y;
  }
  cudaMemcpyToSymbol(const_Mcoor, coor2, sizeof(float2)* SAMPLE_NUM);
  cudaMemcpyToSymbol(const_marker, value, sizeof(float4)* SAMPLE_NUM);
  cudaDeviceSynchronize();
  timer.Pause();
  cout << "CPU: " << timer.get_count() << " ns." << endl;
  // for checking...
  //ofstream outFile1("out.txt");
  //if (!outFile1)
  //  return 0;
  //for (int i = 0; i < SAMPLE_NUM; i++) {
  //  float4 f4 = value[i];
  //  float2 f2 = coor2[i];
  //  outFile1 << f4.x << " " << f4.y << " " << f4.z << " " << f4.w << " ";
  //  outFile1 << f2.x << " " << f2.y << endl;
  //}
  //outFile1.close();
  delete[] coor2;
  delete[] value;
  cudaDeviceSynchronize();
  cout << cudaGetErrorString(cudaGetLastError()) << endl;
  return 0;
}