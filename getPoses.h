#ifndef GETPOSES_H
#define GETPOSES_H

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/count.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>

struct isLessTest { 
    __host__ __device__ 
    bool operator()(const thrust::tuple<float4, float2, bool>& a ) { 
      return (thrust::get<2>(a) == false); 
    };
};

__global__
void isLess_kernel(bool* isEasLess, float* Eas, const float threshold, const int numPoses) {
  const int tIdx = threadIdx.x;
  const int Idx = blockIdx.x * 256 + tIdx;
  
  if (Idx >= numPoses)
    return;
  
  isEasLess[Idx] = (Eas[Idx] < threshold)? true : false;
};

thrust::device_vector<float>::iterator findMin(thrust::device_vector<float>* Eas) {
  return thrust::min_element(thrust::device, Eas->begin(), Eas->end());
};

bool getPoses(thrust::device_vector<float4>* Poses4, thrust::device_vector<float2>* Poses2, 
              thrust::device_vector<float>* Eas, float minEa, const float& delta, int* numPoses) {
  
  // get initial threhold
  const float thresh = 0.1869 * delta + 0.0161 - 0.002;
  minEa += thresh;

  // count reductions
  bool tooHighPercentage = false;
  bool first = true;
  int count = INT_MAX;
  thrust::device_vector<bool> isEasLess(*numPoses, false);
  const int BLOCK_NUM = (*numPoses - 1) / 256 + 1;
  while (true) {
    isLess_kernel <<< BLOCK_NUM, 256 >>> (thrust::raw_pointer_cast(isEasLess.data()), thrust::raw_pointer_cast(Eas->data()), minEa, Eas->size());
    count = thrust::count(thrust::device, isEasLess.begin(), isEasLess.end(), true);
    if (first)
      tooHighPercentage = (count / *numPoses > 0.1);
    if (count < 27000) {
      // cut poses4 and poses2
      typedef thrust::tuple< thrust::device_vector< float4 >::iterator, thrust::device_vector< float2 >::iterator, thrust::device_vector< bool >::iterator > TupleIt;
      typedef thrust::zip_iterator< TupleIt >  ZipIt;
      ZipIt Zend = thrust::remove_if(
        thrust::make_zip_iterator(thrust::make_tuple(Poses4->begin(), Poses2->begin(), isEasLess.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(Poses4->end(), Poses2->end(), isEasLess.end())),
        isLessTest()
      );         
      Poses4->erase(thrust::get<0>(Zend.get_iterator_tuple()), Poses4->end());
      Poses2->erase(thrust::get<1>(Zend.get_iterator_tuple()), Poses2->end());
      *numPoses = count;
      break;
    }
    minEa *= 0.99;
  }
  return tooHighPercentage;
};

#endif