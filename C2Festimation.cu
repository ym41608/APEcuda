#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "parameter.h"
#include "calEa.h"
#include "getPoses.h"
#include "expandPoses.h"
#include "ranPixels.h"

using namespace cv;
using namespace std;

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

float mean(thrust::host_vector<float> &bestDists) {
  float sum = 0;
  float count = 0;
  for (thrust::host_vector<float>::reverse_iterator it = bestDists.rbegin(); it != bestDists.rend(); it++) {
    sum += *it;
    count++;
    if (count > 2)
      break;
  }
  return sum / count;
}

void C2Festimation(float4* P4, float2* P2, const cuda::GpuMat &marker_d, parameter* para, const bool &photo, const bool &verbose) {
  
  // allocate sample memory
  thrust::device_vector<float2> mCoor(SAMPLE_NUM, make_float2(0, 0));
  thrust::device_vector<float4> mValue(SAMPLE_NUM, make_float4(0, 0, 0, 0));
  randSample(&mCoor, &mValue, marker_d, para->mDim, para->markerDim);
  
  // initialize the net
  int numPoses = 5166396;
  float4 *Pose4 = new float4[numPoses];
  float2 *Pose2 = new float2[numPoses];
  ifstream inFile("poses.txt");
  if (!inFile)
    return;
  for (int i = 0; i < numPoses; i++) {
    inFile >> Pose4[i].x;
    inFile >> Pose4[i].y;
    inFile >> Pose4[i].z;
    inFile >> Pose4[i].w;
    inFile >> Pose2[i].x;
    inFile >> Pose2[i].y;
  }
  inFile.close();
  std::cout << "read poses complete!" << endl;
  thrust::device_vector<float4> Poses4(Pose4, Pose4+numPoses);
  thrust::device_vector<float2> Poses2(Pose2, Pose2 + numPoses);
  thrust::device_vector<float> Eas(numPoses);
  
  
  // start
  Timer timer;
  timer.Reset(); timer.Start();
  const float factor = 1/1.511;
  int level = 0;
  unsigned int BLOCK_NUM = 0;
  int2 c;
  thrust::host_vector<float> bestDists;
  while (true) {
    level++;
    
    // calEa
    if (verbose)
      std::cout << "----- Evaluate Ea, with " << numPoses << " poses -----" << endl;
    BLOCK_NUM = (numPoses - 1) / 256 + 1;
    if (photo)
      int tmp = 0;
    else
      calEa_NP_kernel << < BLOCK_NUM, 256 >> > (thrust::raw_pointer_cast(Poses4.data()), thrust::raw_pointer_cast(Poses2.data()), thrust::raw_pointer_cast(Eas.data()), 
        para->Sf, para->P, para->markerDim, para->iDim, numPoses, SAMPLE_NUM);
    
    // findMin
    thrust::device_vector<float>::iterator iter = thrust::min_element(thrust::device, Eas.begin(), Eas.end());
    float bestEa = *iter;
    if (verbose)
      std::cout << "$$$ bestEa = " << bestEa << endl;
    bestDists.push_back(bestEa);
    
    // terminate?
    if ( (bestEa < 0.005) || ((level > 4) && (bestEa < 0.015)) || ((level > 3) && (bestEa > mean(bestDists))) || (level > 7) ) {
      const int idx = iter - Eas.begin();
      *P4 = Poses4[idx];
      *P2 = Poses2[idx];
      timer.Pause();
      cout << "C2F: " << timer.get_count() << " ns." << endl;
      break;
    }
    
    // getPoses
    bool tooHighPercentage = getPoses(&Poses4, &Poses2, &Eas, bestEa, para->delta, &numPoses);
    
    
    if (photo) {
      c.x = 10000000;
      c.y = 7500000;
    } else {
      c.x = 7500000;
      c.y = 5000000;
    }
 
    // restart?
    if ((level==1) && ((tooHighPercentage && (bestEa > 0.05) && (numPoses < c.x)) || ((bestEa > 0.10) && (numPoses < c.y)) ) ) {
      cout << "gg" << endl;
      level = 0;
      bestDists.clear();
    }
    else {
      if (verbose)
        cout << "##### Continuing!!! prevDelta = " << para->delta << ", newDelta = " << para->delta*factor << endl;
      
      // expandPoses
      expandPoses(&Poses4, &Poses2, factor, para, &numPoses);
      
      if (verbose) {
        cout << "***" << endl << "*** level " << level << endl << "***" << endl;
      }
    }
    
    // re-sample
    randSample(&mCoor, &mValue, marker_d, para->mDim, para->markerDim);
  }
  
  delete []Pose4;
  delete []Pose2;
}

int main() {

  // read image
  Mat marker = imread("marker.png");
  marker.convertTo(marker, CV_32FC3, 1 / 255.0);
  cvtColor(marker, marker, CV_BGR2YCrCb);
  cuda::GpuMat marker_d(marker);
  Mat img = imread("img.png");
  img.convertTo(img, CV_32FC3, 1 / 255.0);
  cvtColor(img, img, CV_BGR2YCrCb);
  
  // initial texture memory
  float4* imgYCrCb = new float4[img.cols*img.rows];
  for (int j = 0; j < img.rows; j++) {
    float* img_j = img.ptr<float>(j);
    for (int i = 0; i < img.cols; i++) {
      imgYCrCb[j*img.cols + i] = make_float4(img_j[3 * i], img_j[3 * i + 1], img_j[3 * i + 2], 0);
    }
  }
  
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
  
  // initial parmeter
  parameter para;
  para.mDim = make_int2(marker.cols, marker.rows);
  para.markerDim = make_float2(0.5 * marker.cols / marker.rows, 0.5);
  para.iDim = make_int2(img.cols, img.rows);
  para.delta = 0.25;
  para.Sf = make_float2(1000, -1000);
  para.P = make_int2(400, 300);
  para.s4.x = 0.036084;
  para.s4.y = 0.036084;
  para.s4.z = 0.036084;
  para.s4.w = 0.036084;
  para.s2.x = 0.072169;
  para.s2.y = 0.072169;
  para.Btx = make_float2(-2.7, 2.7);
  para.Bty = make_float2(-1.9, 1.9);
  para.Btz = make_float2(3, 8);
  para.Brx = make_float2(0, 1.3963);
  para.Brz = make_float2(-3.1416, 3.1416);
  
  //
  float4 P4;
  float2 P2;
  C2Festimation(&P4, &P2, marker_d, &para, false, true);
  cout << P4.x << "," << P4.y << "," << P4.z << "," << P4.w << "," << P2.x << "," << P2.y << endl;
  
  delete[] imgYCrCb;
  cudaDeviceSynchronize();
  cout << cudaGetErrorString(cudaGetLastError()) << endl;
  return 0;
}