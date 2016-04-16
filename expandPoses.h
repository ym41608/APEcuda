#ifndef EXPANDPOSES_H
#define EXPANDPOSES_H

#include "parameter.h"
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/count.h>

// for gen random vector 2
struct int2prg {
  __host__ __device__
  int2 operator()(const int n) const {
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> dist(-2, 1);
    rng.discard(n);
    return make_int2(dist(rng), dist(rng));
  }
};

// for gen random vector 4
struct int4prg {
  __host__ __device__
  int4 operator()(const int n) const {
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> dist(-2, 1);
    rng.discard(n);
    return make_int4(dist(rng), dist(rng), dist(rng), dist(rng));
  }
};

struct isValidTest { 
    __host__ __device__ 
    bool operator()(const thrust::tuple<float4, float2, bool>& a ) { 
      return (thrust::get<2>(a) == false); 
    };
};

__global__
void expand_kernel(float4* Poses4, float2* Poses2, const int numPoses, const int newSize) {
  const int tIdx = threadIdx.x;
  const int Idx = blockIdx.x * BLOCK_SIZE + tIdx;
  if (Idx >= numPoses)
    return;
  
  for (int i = Idx + numPoses; i < newSize; i += numPoses) {
    Poses4[i] = Poses4[Idx];
    Poses2[i] = Poses2[Idx];
  }
};

__global__
void add_kernel(float4* Poses4, float2* Poses2, int4* rand4, int2* rand2, bool* isValid, 
                const float4 s4, const float2 s2, const float2 btz, const float2 brx, const float2 marker, const int numPoses, const int expandSize) {
  const int tIdx = threadIdx.x;
  const int Idx = blockIdx.x * BLOCK_SIZE + tIdx;
  if (Idx >= expandSize)
    return;
  float isPlus;
  
  // mem
  float Otx = Poses4[Idx + numPoses].x;
  float Oty = Poses4[Idx + numPoses].y;
  float Otz = Poses4[Idx + numPoses].z;
  float Orx = Poses4[Idx + numPoses].w;
  float Orz0 = Poses2[Idx + numPoses].x;
  float Orz1 = Poses2[Idx + numPoses].y;
  
  // tx ty
  float weight = Otz + sqrtf(marker.x*marker.x + marker.y*marker.y) * sinf(Orx);
  Poses4[Idx + numPoses].x = Otx + float(rand4[Idx].x) * weight * s4.x;
  Poses4[Idx + numPoses].y = Oty + float(rand4[Idx].y) * weight * s4.y;
  
  // tz
  isPlus = float(rand4[Idx].z);
  float vtz = 1 - isPlus * s4.z * Otz;
  Poses4[Idx + numPoses].z = Otz + isPlus * s4.z * (Otz * Otz) / vtz;
  
  // rx
  isPlus = float(rand4[Idx].w);
  float sinrx = 2 - 1/(1/(2 - sinf(Orx)) + isPlus*s4.w);
  Poses4[Idx + numPoses].w = Orx + isPlus * isPlus * (asinf(sinrx) - Orx);
  
  // rz0 rz1
  weight = sqrtf(btz.x * btz.y);
  Poses2[Idx + numPoses].x = Orz0 + float(rand2[Idx].x)*s2.x*weight;
  Poses2[Idx + numPoses].y = Orz1 + float(rand2[Idx].y)*s2.y*weight;
  
  // condition
  isValid[Idx + numPoses] = (vtz != 0) & (abs(sinrx) <= 1) & (Poses4[Idx + numPoses].z >= btz.x) & (Poses4[Idx + numPoses].z <= btz.y) & (Poses4[Idx + numPoses].w >= brx.x) & (Poses4[Idx + numPoses].w <= brx.y);
};

void randVector(thrust::device_vector<int4>* rand4, thrust::device_vector<int2>* rand2, const int& num) {
  thrust::counting_iterator<int> i04(0);
  thrust::counting_iterator<int> i02(22);
  thrust::transform(i04, i04 + num, rand4->begin(), int4prg());
  thrust::transform(i02, i02 + num, rand2->begin(), int2prg());
};

void expandPoses(thrust::device_vector<float4>* Poses4, thrust::device_vector<float2>* Poses2, 
                 const float& factor, parameter* para, int* numPoses) {
  // number of expand points
  const int numPoints = 80;
  int expandSize = (*numPoses) * numPoints;
  int newSize = (*numPoses) * (numPoints + 1);
  
  // decrease step
  para->shrinkNet(factor);
  
  // gen random set
  thrust::device_vector<int4> rand4(expandSize);
  thrust::device_vector<int2> rand2(expandSize);
  randVector(&rand4, &rand2, expandSize);
  
  // expand origin set
  const int BLOCK_NUM0 = ((*numPoses) - 1) / BLOCK_SIZE + 1;
  Poses4->resize(newSize);
  Poses2->resize(newSize);
  expand_kernel << < BLOCK_NUM0, BLOCK_SIZE >> > (thrust::raw_pointer_cast(Poses4->data()), thrust::raw_pointer_cast(Poses2->data()), *numPoses, newSize);

  // add finer delta
  const int BLOCK_NUM1 = (expandSize - 1) / BLOCK_SIZE + 1;
  thrust::device_vector<bool> isValid(newSize, true);
  add_kernel <<< BLOCK_NUM1, BLOCK_SIZE >>> (thrust::raw_pointer_cast(Poses4->data()), thrust::raw_pointer_cast(Poses2->data()), 
                                      thrust::raw_pointer_cast(rand4.data()), thrust::raw_pointer_cast(rand2.data()), 
                                      thrust::raw_pointer_cast(isValid.data()), para->s4, para->s2, 
                                      para->Btz, para->Brx, para->markerDim, *numPoses, expandSize);
  
  // remove invalid
  typedef thrust::tuple< thrust::device_vector< float4 >::iterator, thrust::device_vector< float2 >::iterator, thrust::device_vector< bool >::iterator > TupleIt;
  typedef thrust::zip_iterator< TupleIt >  ZipIt;
  ZipIt Zend = thrust::remove_if(
    thrust::make_zip_iterator(thrust::make_tuple(Poses4->begin(), Poses2->begin(), isValid.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(Poses4->end(), Poses2->end(), isValid.end())),
    isValidTest()
  );         
  Poses4->erase(thrust::get<0>(Zend.get_iterator_tuple()), Poses4->end());
  Poses2->erase(thrust::get<1>(Zend.get_iterator_tuple()), Poses2->end());
  *numPoses = Poses4->size();
};

#endif