#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <chrono>
#include <fstream>
#include "parameter.h"

#define _USE_MATH_DEFINES
#include <math.h>

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

__global__
void createSet_kernel(float4* Poses4, float2* Poses2, const int start, const int4 num, const int numPose, const float tz, const float rx, 
                      const float rzMin, const float tx_w, const float ty_w, const float4 s4, const float2 s2, const float length, const float tz_mid) {
  const int tIdx = threadIdx.x;
  const int Idx = blockIdx.x * BLOCK_SIZE + tIdx;
  
  if (Idx >= numPose)
    return;
  
  const int nrz1 = num.x;
  const int ntx = num.y;
  const int nty = num.z;
  const int nrz0 = num.w;
  
  const int idrz0 = Idx % nrz0;
  const int idty = (Idx / nrz0) % nty;
  const int idtx = (Idx / (nrz0 * nty)) % ntx;
  const int idrz1 = (Idx / (nrz0 * nty * ntx)) % nrz1;
  
  float4 p4;
  float2 p2;
  p4.x = -tx_w + idtx*s4.x*(tz + length*sinf(rx));
  p4.y = -ty_w + idty*s4.y*(tz + length*sinf(rx));
  p4.z = tz;
  p4.w = -rx;
  p2.x = rzMin + idrz0*s2.x*tz_mid;
  p2.y = rzMin + idrz1*s2.y*tz_mid;
  
  Poses4[Idx + start] = p4;
  Poses2[Idx + start] = p2;			
}

void createSet(thrust::device_vector<float4> *Poses4, thrust::device_vector<float2> *Poses2, const parameter &para) {
  
  // count
  int countTotal = 0;
  thrust::host_vector<int4> count; // rz0 rz1 tx ty
   
  // paramters
  const float length = sqrt(para.markerDim.x*para.markerDim.x + para.markerDim.y*para.markerDim.y);
  const float tz_mid = sqrt(para.Btz.x*para.Btz.y);
  const int numRz0 = int((para.Brz.y - para.Brz.x) / (para.s2.x*tz_mid)) + 1;
  const int numRz1 = int((para.Brz.y - para.Brz.x) / (para.s2.y*tz_mid)) + 1;
  
  // counting
  for (float tz = para.Btz.x; tz <= para.Btz.y; ) {
		float tx_w = fabs(para.P.x*tz / para.Sf.x - para.markerDim.y);
		float ty_w = fabs(para.P.y*tz / fabs(para.Sf.y) - para.markerDim.y);
    for (float rx = para.Brx.x; rx >= -para.Brx.y; ) {
      int nrz0 = (rx != 0)? numRz0:1;
      int ntx = int(2*tx_w / (para.s4.x*(tz + length*sin(rx)))) + 1;
      int nty = int(2*ty_w / (para.s4.y*(tz + length*sin(rx)))) + 1;
      countTotal += (nrz0 * numRz1 * ntx * nty);
      count.push_back(make_int4(numRz1, ntx, nty, nrz0));
      
      double sinValuey = 1 / (1/(2+sin(rx)) + para.s4.w) - 2;
      if (sinValuey <= 1 && sinValuey >= -1)
        rx = asin(sinValuey);
      else
        rx = -para.Brx.y - 1;
    }
    tz += tz*tz*para.s4.z / (1 - para.s4.z*tz);
  }
  
  // allocate
		cout <<count.size() << endl;
  cout << "count ready! " << countTotal << endl;
  Poses4->resize(countTotal);
  Poses2->resize(countTotal);
  
  // assignment
  float4* Poses4ptr = thrust::raw_pointer_cast(Poses4->data());
  float2* Poses2ptr = thrust::raw_pointer_cast(Poses2->data());
  thrust::host_vector<int4>::iterator it = count.begin();
  int start = 0;
  for (float tz = para.Btz.x; tz <= para.Btz.y; ) {
		float tx_w = fabs(para.P.x*tz / para.Sf.x - para.markerDim.y);
		float ty_w = fabs(para.P.y*tz / fabs(para.Sf.y) - para.markerDim.y);
    for (float rx = para.Brx.x; rx >= -para.Brx.y; ) {
      int numPose = (*it).x * (*it).y * (*it).z * (*it).w;
			//cout << numPose << "," << (*it).x << "," << (*it).y << "," << (*it).z << "," << (*it).w << endl;
      int BLOCK_NUM = (numPose - 1)/BLOCK_SIZE + 1;
      createSet_kernel <<< BLOCK_NUM, BLOCK_SIZE >>> (Poses4ptr, Poses2ptr, start, *it, numPose, tz, rx, para.Brz.x, tx_w, ty_w, para.s4, para.s2, length, tz_mid);
      start += numPose;
      it++;
      
      double sinValuey = 1 / (1/(2+sin(rx)) + para.s4.w) - 2;
      if (sinValuey <= 1 && sinValuey >= -1)
        rx = asin(sinValuey);
      else
        rx = -para.Brx.y - 1;
    }
    tz += tz*tz*para.s4.z / (1 - para.s4.z*tz);
  }
  
  if (start != countTotal)
    cout << "error orrcur!" << endl;
}

int main() {
  
  parameter para;
  float delta = 0.4;
  
  // dim of images
  para.mDim.x = 640;
  para.mDim.y = 480;
  para.iDim.x = 800;
  para.iDim.y = 600;
  
  // intrinsic parameter
  para.Sf.x = 1000;
  para.Sf.y = -1000;
  para.P = make_int2(400, 300);
  
  // search range in pose domain
  float wm = float(640);
  float hm = float(480);
  float wi = float(800);
  float hi = float(600);
  float minmDim = fmin(hm, wm);
  para.markerDim.x = wm / minmDim * 0.5;
  para.markerDim.y = hm / minmDim * 0.5;
  float minmarkerDim = fmin(para.markerDim.x, para.markerDim.y);
  para.Btz = make_float2(3, 8);
  para.Brx = make_float2(0, 80 * M_PI / 180);
  para.Brz = make_float2(-M_PI, M_PI);
  
  //bounds
  float m_tz = sqrt(24);
  float sqrt2 = sqrt(2);
  para.s4 = make_float4(delta/sqrt2/m_tz, delta/sqrt2/m_tz, delta/sqrt2/m_tz, delta/sqrt2/m_tz);
  para.s2 = make_float2(delta*sqrt2/m_tz, delta*sqrt2/m_tz);
		para.delta = delta;

  thrust::device_vector<float4> Poses4(100);
  thrust::device_vector<float2> Poses2(100);
  Timer timer;
  timer.Reset(); timer.Start();
		createSet(&Poses4, &Poses2, para);
  timer.Pause();
  std::cout << "Time: " << timer.get_count() << " ns." << std::endl;
  
  //ofstream outFile("PoseCuda.txt");
  //if (!outFile)
  //  return 0;
  //for (int i = 0; i < Poses4.size(); i++) {
		//		float4 p4 = Poses4[i];
		//		float2 p2 = Poses2[i];
		//		outFile << p4.x << " " << p4.y << " " << p4.z << " " << p4.w << " ";
		//		outFile << p2.x << " " << p2.y << " " << endl;
  //}
  //outFile.close();
}