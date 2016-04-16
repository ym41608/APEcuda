#ifndef C2F_H
#define C2F_H

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include "parameter.h"
#include "calEa.h"
#include "getPoses.h"
#include "expandPoses.h"
#include "ranPixels.h"

using namespace cv;
using namespace std;

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
		float ty_w = fabs(para.P.y*tz / para.Sf.y - para.markerDim.y);
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
  Poses4->resize(countTotal);
  Poses2->resize(countTotal);
  
  // assignment
  float4* Poses4ptr = thrust::raw_pointer_cast(Poses4->data());
  float2* Poses2ptr = thrust::raw_pointer_cast(Poses2->data());
  thrust::host_vector<int4>::iterator it = count.begin();
  int start = 0;
  for (float tz = para.Btz.x; tz <= para.Btz.y; ) {
		float tx_w = fabs(para.P.x*tz / para.Sf.x - para.markerDim.y);
		float ty_w = fabs(para.P.y*tz / para.Sf.y - para.markerDim.y);
    for (float rx = para.Brx.x; rx >= -para.Brx.y; ) {
      int numPose = (*it).x * (*it).y * (*it).z * (*it).w;
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

void C2Festimate(float4* P4, float2* P2, const gpu::GpuMat &marker_d, parameter* para, const bool &photo, const bool &verbose) {
  
  // allocate sample memory
  thrust::device_vector<float2> mCoor(SAMPLE_NUM, make_float2(0, 0));
  thrust::device_vector<float4> mValue(SAMPLE_NUM, make_float4(0, 0, 0, 0));
  randSample(&mCoor, &mValue, marker_d, para->mDim, para->markerDim);
  
  // initialize the net
  thrust::device_vector<float4> Poses4;
  thrust::device_vector<float2> Poses2;
  createSet(&Poses4, &Poses2, *para);
  int numPoses = Poses4.size();
  thrust::device_vector<float> Eas(numPoses);
  
  // start
  const float factor = 1/1.511;
  int level = 0;
  unsigned int BLOCK_NUM = 0;
  int2 c;
  thrust::host_vector<float> bestDists;
  while (true) {
    level++;
    if (verbose)
      cout << endl << "***" << endl << "*** level " << level << endl << "***" << endl;
    
    // calEa
    if (verbose)
      cout << "----- Evaluate Ea, with " << numPoses << " poses -----" << endl;
    Eas.resize(numPoses);
    BLOCK_NUM = (numPoses - 1) / BLOCK_SIZE + 1;
    if (photo) {
      calEa_P_kernel << < BLOCK_NUM, BLOCK_SIZE >> > (thrust::raw_pointer_cast(Poses4.data()), thrust::raw_pointer_cast(Poses2.data()), thrust::raw_pointer_cast(Eas.data()), 
        para->Sf, para->P, para->markerDim, para->iDim, numPoses);
    }
    else
      calEa_NP_kernel << < BLOCK_NUM, BLOCK_SIZE >> > (thrust::raw_pointer_cast(Poses4.data()), thrust::raw_pointer_cast(Poses2.data()), thrust::raw_pointer_cast(Eas.data()), 
        para->Sf, para->P, para->markerDim, para->iDim, numPoses);
    
    // findMin
    thrust::device_vector<float>::iterator iter = thrust::min_element(thrust::device, Eas.begin(), Eas.end());
    float bestEa = *iter;
    if (verbose)
      std::cout << "$$$ bestEa = " << bestEa << endl;
    bestDists.push_back(bestEa);
    
    // terminate
    if ( (bestEa < 0.005) || ((level > 4) && (bestEa < 0.015)) || ((level > 3) && (bestEa > mean(bestDists))) || (level > 7) ) {
      const int idx = iter - Eas.begin();
      *P4 = Poses4[idx];
      *P2 = Poses2[idx];
      break;
    }
    
    // getPoses
    bool tooHighPercentage = getPoses(&Poses4, &Poses2, &Eas, bestEa, para->delta, &numPoses);
    
    // restart?
    if (photo) {
      c.x = 10000000;
      c.y = 7500000;
    } 
    else {
      c.x = 7500000;
      c.y = 5000000;
    }
    if ((level==1) && ((tooHighPercentage && (bestEa > 0.05) && (numPoses < c.x)) || ((bestEa > 0.10) && (numPoses < c.y)) ) ) {
      if (verbose)
        cout << "##### Restarting!!! change delta from " << para->delta << " to " << para->delta*0.9 << endl;
      para->shrinkNet(0.9);
      createSet(&Poses4, &Poses2, *para);
      numPoses = Poses4.size();
      level = 0;
      bestDists.clear();
    }
    else {
    // expandPoses
    expandPoses(&Poses4, &Poses2, factor, para, &numPoses);
    if (verbose)
      cout << "##### Continuing!!! prevDelta = " << para->delta << ", newDelta = " << para->delta*factor << endl;
    }
    
    // re-sample
    randSample(&mCoor, &mValue, marker_d, para->mDim, para->markerDim);
  }
}

#endif