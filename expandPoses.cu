#include <iostream>
#include <fstream>
#include <chrono>
#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/count.h>

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

struct poseStep {
  float4 s4;
  float2 s2;
};

struct poseBound {
  float2 tx;
  float2 ty;
  float2 tz;
  float2 rx;
  float2 rz0;
  float2 rz1;
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
  const int Idx = blockIdx.x * 256 + tIdx;
  if (Idx >= numPoses)
    return;
  
  for (int i = Idx + numPoses; i < newSize; i += numPoses) {
    Poses4[i] = Poses4[Idx];
    Poses2[i] = Poses2[Idx];
  }
}

__global__
void add_kernel(float4* Poses4, float2* Poses2, int4* rand4, int2* rand2, bool* isValid, 
                const float4 s4, const float2 s2, const float2 btx, const float2 bty, const float2 btz, 
                const float2 brx, const float2 brz0, const float2 brz1, const float2 marker, const int numPoses, const int expandSize) {
  const int tIdx = threadIdx.x;
  const int Idx = blockIdx.x * 256 + tIdx;
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
}

void randVector(thrust::device_vector<int4>* rand4, thrust::device_vector<int2>* rand2, const int& num) {
  thrust::counting_iterator<int> i04(0);
  thrust::counting_iterator<int> i02(22);
  thrust::transform(i04, i04 + num, rand4->begin(), int4prg());
  thrust::transform(i02, i02 + num, rand2->begin(), int2prg());
}

void expandPoses(thrust::device_vector<float4>* Poses4, thrust::device_vector<float2>* Poses2, 
                 const float& factor, poseStep* step, const poseBound bound, const float2 marker, int* numPoses) {
  // number of expand points
  const int numPoints = 80;
  int expandSize = (*numPoses) * numPoints;
  int newSize = (*numPoses) * (numPoints + 1);
  
  // decrease step
  step->s4.x /= factor;
  step->s4.y /= factor;
  step->s4.z /= factor;
  step->s4.w /= factor;
  step->s2.x /= factor;
  step->s2.y /= factor;
  
  // gen random set
  thrust::device_vector<int4> rand4(expandSize);
  thrust::device_vector<int2> rand2(expandSize);
  randVector(&rand4, &rand2, expandSize);
  
  // expand origin set
  const int BLOCK_NUM0 = ((*numPoses) - 1) / 256 + 1;
  Poses4->resize(newSize);
  Poses2->resize(newSize);
  expand_kernel << < BLOCK_NUM0, 256 >> > (thrust::raw_pointer_cast(Poses4->data()), thrust::raw_pointer_cast(Poses2->data()), *numPoses, newSize);

  // add finer delta
  const int BLOCK_NUM1 = (expandSize - 1) / 256 + 1;
  thrust::device_vector<bool> isValid(newSize, true);
  add_kernel <<< BLOCK_NUM1, 256 >>> (thrust::raw_pointer_cast(Poses4->data()), thrust::raw_pointer_cast(Poses2->data()), 
                                      thrust::raw_pointer_cast(rand4.data()), thrust::raw_pointer_cast(rand2.data()), 
                                      thrust::raw_pointer_cast(isValid.data()), step->s4, step->s2, 
                                      bound.tx, bound.ty, bound.tz, bound.rx, bound.rz0, bound.rz1, marker, *numPoses, expandSize);
  
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
}

int main() {
	
  // read poses
  const float delta = 0.25;
  int numPoses = 25887;
  int numPoints = 80;
  float4 *Pose4 = new float4[numPoses];
  float2 *Pose2 = new float2[numPoses];
  int4 *rand4 = new int4[numPoses*numPoints];
  int2 *rand2 = new int2[numPoses*numPoints];
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

  // read rand
  //ifstream inFile1("rand.txt");
  //if (!inFile1)
  //  return 0;
  //for (int i = 0; i < numPoses*numPoints; i++) {
  //  float tmp;
  //  inFile1 >> tmp;
  //  rand4[i].x = int(tmp);
  //  inFile1 >> tmp;
  //  rand4[i].y = int(tmp);
  //  inFile1 >> tmp;
  //  rand4[i].z = int(tmp);
  //  inFile1 >> tmp;
  //  rand4[i].w = int(tmp);
  //  inFile1 >> tmp;
  //  rand2[i].x = int(tmp);
  //  inFile1 >> tmp;
  //  rand2[i].y = int(tmp);
  //}
  //inFile1.close();
  //cout << "read rand complete!" << endl;
  cout << "original " << numPoses << " poses." << endl;
  
  // create parameter
  poseStep step;
  poseBound bound;
  float2 marker;
  step.s4.x = 0.036084;
  step.s4.y = 0.036084;
  step.s4.z = 0.036084;
  step.s4.w = 0.036084;
  step.s2.x = 0.072169;
  step.s2.y = 0.072169;
  bound.tx = make_float2(-2.7, 2.7);
  bound.ty = make_float2(-1.9, 1.9);
  bound.tz = make_float2(3, 8);
  bound.rx = make_float2(0, 1.3963);
  bound.rz0 = make_float2(-3.1416, 3.1416);
  bound.rz1 = make_float2(-3.1416, 3.1416);
  marker.x = 0.6667;
  marker.y = 0.5;
  
  // load to gpu
  thrust::device_vector<float4> Poses4(Pose4, Pose4 + numPoses);
  thrust::device_vector<float2> Poses2(Pose2, Pose2 + numPoses);
  thrust::device_vector<int4> rands4(rand4, rand4 + numPoses*numPoints);
  thrust::device_vector<int2> rands2(rand2, rand2 + numPoses*numPoints);
  Timer timer;
  timer.Reset(); timer.Start();
  expandPoses(&Poses4, &Poses2, 1.511, &step, bound, marker, &numPoses);
  //expandPoses(&Poses4, &Poses2, 1.511, &step, bound, marker, &numPoses, &rands4, &rands2);
  cout << cudaGetErrorString(cudaGetLastError()) << endl;
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
  delete[] rand4;
  delete[] rand2;
  return 0;
}
















