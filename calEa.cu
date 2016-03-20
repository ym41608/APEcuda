#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <thrust/device_vector.h>

using namespace cv;
using namespace std;

// Block Size
static unsigned int BLOCK_SIZE = 256;

// Texture
texture<float4, cudaTextureType2D, cudaReadModeElementType> tex_imgYCrCb;

// Constant
__constant__ float2 const_Mcoor[444];
__constant__ float4 const_marker[444];

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

__global__ void calEa_kernel(float4 *Poses4, float2 *Poses2, float *Eas, const float2 Sf, const int2 P, const float2 normDim, const int2 imgDim,
  const int numPoses, const int numPoints) {
  const int tIdx = threadIdx.x;
  const int Idx = blockIdx.x * 256 + tIdx;

  if (Idx >= numPoses)
    return;

  // calculate transformation
  float tx, ty, tz, rx, rz0, rz1;
  float rz0Cos, rz0Sin, rz1Cos, rz1Sin, rxCos, rxSin;
  float t0, t1, t3, t4, t5, t7, t8, t9, t11;
  float r11, r12, r21, r22, r31, r32;

  // get pose parameter
  tx = Poses4[Idx].x;
  ty = Poses4[Idx].y;
  tz = Poses4[Idx].z;
  rx = Poses4[Idx].w;
  rz0 = Poses2[Idx].x;
  rz1 = Poses2[Idx].y;

  rz0Cos = cosf(rz0); rz0Sin = sinf(rz0);
  rz1Cos = cosf(rz1); rz1Sin = sinf(rz1);
  rxCos = cosf(rx); rxSin = sinf(rx);

  //  z coordinate is y cross x   so add minus
  r11 = rz0Cos * rz1Cos - rz0Sin * rxCos * rz1Sin;
  r12 = -rz0Cos * rz1Sin - rz0Sin * rxCos * rz1Cos;
  r21 = rz0Sin * rz1Cos + rz0Cos * rxCos * rz1Sin;
  r22 = -rz0Sin * rz1Sin + rz0Cos * rxCos * rz1Cos;
  r31 = rxSin * rz1Sin;
  r32 = rxSin * rz1Cos;

  // final transfomration
  t0 = Sf.x*r11 + P.x*r31;
  t1 = Sf.x*r12 + P.x*r32;
  t3 = Sf.x*tx + P.x*tz;
  t4 = Sf.y*r21 + (P.y - 1)*r31;
  t5 = Sf.y*r22 + (P.y - 1)*r32;
  t7 = Sf.y*ty + (P.y - 1)*tz;
  t8 = r31;
  t9 = r32;
  t11 = tz;

  // reject transformations make marker out of boundary
  float invc1z = 1 / (t8*(-normDim.x) + t9*(-normDim.y) + t11);
  float c1x = (t0*(-normDim.x) + t1*(-normDim.y) + t3) * invc1z;
  float c1y = (t4*(-normDim.x) + t5*(-normDim.y) + t7) * invc1z;
  float invc2z = 1 / (t8*(+normDim.x) + t9*(-normDim.y) + t11);
  float c2x = (t0*(+normDim.x) + t1*(-normDim.y) + t3) * invc2z;
  float c2y = (t4*(+normDim.x) + t5*(-normDim.y) + t7) * invc2z;
  float invc3z = 1 / (t8*(+normDim.x) + t9*(+normDim.y) + t11);
  float c3x = (t0*(+normDim.x) + t1*(+normDim.y) + t3) * invc3z;
  float c3y = (t4*(+normDim.x) + t5*(+normDim.y) + t7) * invc3z;
  float invc4z = 1 / (t8*(-normDim.x) + t9*(+normDim.y) + t11);
  float c4x = (t0*(-normDim.x) + t1*(+normDim.y) + t3) * invc4z;
  float c4y = (t4*(-normDim.x) + t5*(+normDim.y) + t7) * invc4z;
  float minx = min(c1x, min(c2x, min(c3x, c4x)));
  float maxx = max(c1x, max(c2x, max(c3x, c4x)));
  float miny = min(c1y, min(c2y, min(c3y, c4y)));
  float maxy = max(c1y, max(c2y, max(c3y, c4y)));
  if ((minx < 0) | (maxx >= imgDim.x) | (miny < 0) | (maxy >= imgDim.y)) {
    Eas[Idx] = FLT_MAX;
    return;
  }

  // calculate Ea
  float score = 0.0;
  float invz;
  float4 YCrCb_tex, YCrCb_const;
  float u, v;
  for (int i = 0; i < numPoints; i++) {
    
    // calculate coordinate on camera image
    invz = 1 / (t8*const_Mcoor[i].x + t9*const_Mcoor[i].y + t11);
    u = (t0*const_Mcoor[i].x + t1*const_Mcoor[i].y + t3) * invz;
    v = (t4*const_Mcoor[i].x + t5*const_Mcoor[i].y + t7) * invz;

    // get value from constmem
    YCrCb_const = const_marker[i];

    // get value from texture
    YCrCb_tex = tex2D(tex_imgYCrCb, u, v);

    // calculate distant
    score += (2 * abs(YCrCb_tex.x - YCrCb_const.x) + abs(YCrCb_tex.y - YCrCb_const.y) + abs(YCrCb_tex.z - YCrCb_const.z));
  }
  Eas[Idx] = score / (numPoints * 4);
}


void calEa(float* Eas, float4* Pose4, float2* Pose2, const float& Sxf, const float& Syf, const int& x_w, const int& y_h,
  const double& marker_w, const double& marker_h, const int &wI, const int& hI, const int& numPoses) {

  // copy poses and Eas to device memory
  thrust::device_vector<float4> Poses4(Pose4, Pose4 + numPoses);
  thrust::device_vector<float2> Poses2(Pose2, Pose2 + numPoses);
  thrust::device_vector<float> Eas_d(Eas, Eas + numPoses);

  // CUDA Kernel
  Timer timer;
  timer.Reset(); timer.Start();
  unsigned int BLOCK_NUM = (numPoses - 1) / 256 + 1;
  calEa_kernel << < BLOCK_NUM, 256 >> > (thrust::raw_pointer_cast(Poses4.data()), thrust::raw_pointer_cast(Poses2.data()), thrust::raw_pointer_cast(Eas_d.data()), 
    make_float2(Sxf, Syf), make_int2(x_w, y_h), make_float2(marker_w, marker_h), make_int2(wI, hI), numPoses, 444);
  if (cudaDeviceSynchronize() != cudaSuccess) {
    cerr << "error in kernel: " << cudaGetErrorString(cudaGetLastError()) << endl;
    exit(-1);
  }
  timer.Pause();
  cout << "Kernel: " << timer.get_count() << " ns." << endl;

  // copy and free
  cudaMemcpy(Eas, thrust::raw_pointer_cast(Eas_d.data()), numPoses * sizeof(float), cudaMemcpyDeviceToHost);
}

int main() {

  // read image
  Mat img = imread("img.png");
  img.convertTo(img, CV_32FC3, 1 / 255.0);
  cvtColor(img, img, CV_BGR2YCrCb);
  Mat marker = imread("marker.png");
  marker.convertTo(marker, CV_32FC3, 1 / 255.0);
  cvtColor(marker, marker, CV_BGR2YCrCb);

  // read poses
  const int numPoses = 5166396;
  float4 *Pose4 = new float4[numPoses];
  float2 *Pose2 = new float2[numPoses];
  float *Eas = new float[numPoses];
  ifstream inFile("poses.txt");
  if (!inFile)
    return 0;
  for (int i = 0; i < numPoses; i++) {
    inFile >> Pose4[i].x;
    inFile >> Pose4[i].y;
    inFile >> Pose4[i].z;
    inFile >> Pose4[i].w;
    inFile >> Pose2[i].x;
    inFile >> Pose2[i].y;
  }
  inFile.close();
  cout << "read poses complete!" << endl;
  
  // get random points
  float marker_w = 0.5 * marker.cols / marker.rows;
  float marker_h = 0.5;

  // initial constant memory
  float2 *coor2 = new float2[444];
  float4 *value = new float4[444];
  for (int i = 0; i < 444; i++) {
    int x = rand() % marker.cols;
    int y = rand() % marker.rows;
    Vec3f YCrCb = marker.at<Vec3f>(y, x);
    value[i] = make_float4(YCrCb[0], YCrCb[1], YCrCb[2], 0);
    coor2[i].x = (2 * float(x) - marker.cols) / marker.cols * marker_w;
    coor2[i].y = -(2 * float(y) - marker.rows) / marker.rows * marker_h;
  }

  // initial texture memory
  float4* imgYCrCb = new float4[img.cols*img.rows];
  for (int j = 0; j < img.rows; j++) {
    float* img_j = img.ptr<float>(j);
    for (int i = 0; i < img.cols; i++) {
      imgYCrCb[j*img.cols + i] = make_float4(img_j[3 * i], img_j[3 * i + 1], img_j[3 * i + 2], 0);
    }
  }
  Timer timer;
  timer.Reset(); timer.Start();
  cudaMemcpyToSymbol(const_Mcoor, coor2, sizeof(float2)* 444);
  cudaMemcpyToSymbol(const_marker, value, sizeof(float4)* 444);

  // set texture parameters
  tex_imgYCrCb.addressMode[0] = cudaAddressModeBorder;
  tex_imgYCrCb.addressMode[1] = cudaAddressModeBorder;
  tex_imgYCrCb.filterMode = cudaFilterModePoint;
  tex_imgYCrCb.normalized = false;

  // copy to texture memory
  cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
  cudaArray *imgYCrCbArray;
  cudaMallocArray(&imgYCrCbArray, &desc, img.cols, img.rows);
  cudaMemcpyToArray(imgYCrCbArray, 0, 0, imgYCrCb, sizeof(float4)*img.cols*img.rows, cudaMemcpyHostToDevice);
  cudaBindTextureToArray(tex_imgYCrCb, imgYCrCbArray, desc);

  calEa(Eas, Pose4, Pose2, 1000, -1000, 400, 300, marker_w, marker_h, img.cols, img.rows, numPoses);
  timer.Pause();
  cout << "Overall: " << timer.get_count() << " ns." << endl;
  ofstream outFile("EasCuda.txt");
  if (!outFile)
    return 0;
  for (int i = 0; i < numPoses; i++) {
    outFile << Eas[i] << endl;
  }
  outFile.close();
  
  delete[] Pose4;
  delete[] Pose2;
  delete[] coor2;
  delete[] value;
  delete[] Eas;
  delete[] imgYCrCb;
  cudaUnbindTexture(tex_imgYCrCb);
  cudaFreeArray(imgYCrCbArray);
  cout << cudaGetErrorString(cudaGetLastError()) << endl;
  return 0;
}