#ifndef APE_H
#define APE_H

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <thrust/device_vector.h>
#include "parameter.h"
#include "C2Festimate.h"
#include "preCal.h"

using namespace cv;

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

void APE(float4 *P4, float2 *P2, const Mat &marker, const Mat &img, const float2 &Sf, const int2 &P, const bool &photo, const float2 &Btz, const bool &verbose) {
  
  Timer timer;
  timer.Reset(); timer.Start();
  // allocate
  parameter para;
  gpu::GpuMat marker_d(marker.cols, marker.rows, CV_32FC3);
  gpu::GpuMat img_d(marker.cols, marker.rows, CV_32FC4);

  // pre-calculation
  preCal(para, marker_d, img_d, marker, img, Sf, P, 0.25, Btz, verbose);
  timer.Pause();
  cout << "Pre-Time: " << timer.get_count() << " ns." << endl;
  
  // C2Festimation
  C2Festimate(P4, P2, marker_d, &para, photo, verbose);
};

#endif