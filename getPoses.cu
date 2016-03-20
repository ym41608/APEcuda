#include <iostream>
#include <fstream>
#include <chrono>
#include <math.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/count.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>

using namespace std;

static const int BLOCK_SIZE = 256;

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
}

void getPoses(thrust::device_vector<float4>* Poses4, thrust::device_vector<float2>* Poses2, 
              thrust::device_vector<float>* Eas, const float& delta, int* numPoses) {
  
  // get initial threhold
  const float thresh = 0.1869 * delta + 0.0161 - 0.002;
  thrust::device_vector<float>::iterator iter = thrust::min_element(thrust::device, Eas->begin(), Eas->end());
  float minEa = *iter + thresh;

  // count reductions
  int count = INT_MAX;
  thrust::device_vector<bool> isEasLess(Eas->size(), false);
  const int BLOCK_NUM = (*numPoses - 1) / 256 + 1;
  while (true) {
    isLess_kernel <<< BLOCK_NUM, 256 >>> (thrust::raw_pointer_cast(isEasLess.data()), thrust::raw_pointer_cast(Eas->data()), minEa, Eas->size());
    count = thrust::count(thrust::device, isEasLess.begin(), isEasLess.end(), true);
    if (count < 27000) {
      // cut poses4 and poses2
      Timer timer;
      timer.Reset(); timer.Start();
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
      timer.Pause();
      cout << "Cutting Time: " << timer.get_count() << " ns." << endl;
      break;
    }
    minEa *= 0.99;
  }
}

int main() {
	
  
  // read poses
  const float delta = 0.25;
  int numPoses = 5166396;
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
  cout << "read pose complete!" << endl;
  
  // read Eas
  ifstream inFile1("Eas.txt");
  if (!inFile1)
    return 0;
  for (int i = 0; i < numPoses; i++) {
    inFile1 >> Eas[i];
  }
  inFile1.close();
  cout << "read Ea complete!" << endl;
  cout << "original " << numPoses << " poses." << endl;
  thrust::device_vector<float4> Poses4(numPoses);
  thrust::device_vector<float2> Poses2(numPoses);
  thrust::copy(Pose4, Pose4 + numPoses, Poses4.begin());
  thrust::copy(Pose2, Pose2 + numPoses, Poses2.begin());
  thrust::device_vector<float> Eas_d(Eas, Eas + numPoses);
  Timer timer;
  timer.Reset(); timer.Start();
  getPoses(&Poses4, &Poses2, &Eas_d, delta, &numPoses);
  timer.Pause();
  cout << "Time: " << timer.get_count() << " ns." << endl;
  cout << "now " << numPoses << " poses." << endl;
  
  ofstream outFile("poses1Cuda.txt");
  if (!outFile)
    return 0;
  for (int i = 0; i < numPoses; i++) {
    float4 p4 = Poses4[i];
    float2 p2 = Poses2[i];
    outFile << p4.x << " " << p4.y << " " << p4.z << " " << p4.w << " ";
    outFile << p2.x << " " << p2.y << endl;
  }
  outFile.close();
  
  delete[] Pose4;
  delete[] Pose2;
  delete[] Eas;
  cout << cudaGetErrorString(cudaGetLastError()) << endl;
  return 0;
}
















