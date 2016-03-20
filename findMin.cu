#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <math.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>

using namespace cv;
using namespace std;

// Block Size
static unsigned int BLOCK_SIZE = 256;

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

template <unsigned int blockSize> __global__ void findMin_kernel(float *minEa, float *Eas, const unsigned int numPoses) {
  
  //extern __shared__ int sdata[];
  __shared__ float sdata[blockSize];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockSize*2) + tid;
  unsigned int gridSize = blockSize*2*gridDim.x;
  //sdata[tid] = fminf(Eas[i], Eas[i + blockSize]);
  sdata[tid] = INT_MAX;
  while (i  < numPoses) {
    float tmp = fminf(Eas[i], Eas[i+blockSize]);
	sdata[tid] = fminf(sdata[tid], tmp);
	i += gridSize;
  }
  __syncthreads();
  if (tid < 128) { sdata[tid] = fminf(sdata[tid], sdata[tid + 128]); } __syncthreads();
  if (tid <   64) {sdata[tid] = fminf(sdata[tid], sdata[tid + 64]); } __syncthreads();
  if (tid < 32) {
	  sdata[tid] = fminf(sdata[tid], sdata[tid + 32]);
	  sdata[tid] = fminf(sdata[tid], sdata[tid + 16]); 
	  sdata[tid] = fminf(sdata[tid], sdata[tid + 8]); 
	  sdata[tid] = fminf(sdata[tid], sdata[tid + 4]);	 
	  sdata[tid] = fminf(sdata[tid], sdata[tid + 2]);	
	  sdata[tid] = fminf(sdata[tid], sdata[tid + 1]);	 
  }
  if (tid == 0) minEa[blockIdx.x] = sdata[0];

  //if (blockSize >= 512) { if (tid < 256) { sdata[tid] = fminf(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
  //if (blockSize >= 256) { if (tid < 128) { sdata[tid] = fminf(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
  //if (blockSize >= 128) { if (tid <   64) {sdata[tid] = fminf(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
  //if (tid < 32) {
  //if (blockSize >= 64) {sdata[tid] = fminf(sdata[tid], sdata[tid + 32]);__syncthreads();}
  //if (blockSize >= 32) {sdata[tid] = fminf(sdata[tid], sdata[tid + 16]);__syncthreads();}
  //if (blockSize >= 16) {sdata[tid] = fminf(sdata[tid], sdata[tid + 8]);	__syncthreads();}
  //if (blockSize >= 8) {sdata[tid] = fminf(sdata[tid], sdata[tid + 4]);	__syncthreads();}
  //if (blockSize >= 4) {sdata[tid] = fminf(sdata[tid], sdata[tid + 2]);	__syncthreads();}
  //if (blockSize >= 2) {sdata[tid] = fminf(sdata[tid], sdata[tid + 1]);	__syncthreads();}
  //}
  //if (tid == 0) minEa[blockIdx.x] = sdata[0];
}

void findMin(float *minEa, float *Eas, const int &numPoses) {
  
  thrust::device_vector<float> Eas_d(Eas, Eas + numPoses); 
  float* buffer_d;
  cudaMalloc(&buffer_d, numPoses * sizeof(float));
  cudaMemcpy(&buffer_d[0], Eas, numPoses * sizeof(float), cudaMemcpyHostToDevice);
  thrust::device_ptr<float> dev_ptr(buffer_d);
  Timer timer;
  timer.Reset(); timer.Start();
  //thrust::device_vector<float>::iterator iter = thrust::min_element(thrust::device, Eas_d.begin(), Eas_d.end());
  float *iter = thrust::raw_pointer_cast(thrust::min_element(thrust::device, dev_ptr, dev_ptr + numPoses));
  timer.Pause();
  cout << "Overall gpu: " << timer.get_count() << " ns." << endl;
  cudaMemcpy(minEa, iter, sizeof(float), cudaMemcpyDeviceToHost);
  //*minEa = *iter;
}

void findMin_cpu(float *minEa, float *Eas, const int &numPoses) {
	*minEa = INT_MAX;
	for (int i = 0; i < numPoses; i++) {
		if (*minEa > Eas[i])
			*minEa = Eas[i];
	}
  //float *tmp = thrust::min_element(thrust::host, Eas, Eas + numPoses);
  //*minEa = *tmp;
}

int main() {
	
  // read Eas
  const int numPoses = 5166000;
  float *Eas = new float[numPoses];
  float minEa;
  ifstream inFile("EasCuda.txt");
  if (!inFile)
	return 0;
  for (int i = 0; i < numPoses; i++) {
    float tmp;
    inFile >> tmp;
    Eas[i] = tmp;
  }
  


  Timer timer;
  
  

  //timer.Reset(); timer.Start();
  findMin(&minEa, Eas, numPoses);
  //timer.Pause();
  //cout << "Overall gpu: " << timer.get_count() << " ns." << endl;
  
  timer.Reset(); timer.Start();
  float minEa_cpu;
  findMin_cpu(&minEa_cpu, Eas, numPoses);
  timer.Pause();
  cout << "Overall cpu: " << timer.get_count() << " ns." << endl;

  if (minEa_cpu == minEa)
	  cout << "result correct: " << minEa << endl;
  else
	  cout << "gpu: " << minEa << ", cpu: " << minEa_cpu << endl;



  delete[] Eas;
  cout << cudaGetErrorString(cudaGetLastError()) << endl;
  return 0;
}
















