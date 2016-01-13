#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <chrono>

using namespace cv;
using namespace std;

// Block Size
static unsigned int BLOCK_SIZE = 256;

// Texture
texture<float, cudaTextureType2D, cudaReadModeElementType> tex_imgY;
texture<float, cudaTextureType2D, cudaReadModeElementType> tex_imgCr;
texture<float, cudaTextureType2D, cudaReadModeElementType> tex_imgCb;

// Constant
__constant__ float2 const_Mcoor[444];
__constant__ float const_marker[444 * 3];

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

__global__ void calEa_kernel(const float *Poses, float *Eas, const float2 Sf, const int2 P, const float2 normDim, const int2 imgDim,
                             const int numPoses, const int numPoints) {
  const int tIdx = threadIdx.x;
  const int Idx = blockIdx.x * blockDim.x + tIdx;
  
  if (Idx >= numPoses)
    return;
  
  // calculate transformation
  float tx, ty, tz, rx, rz0, rz1;
  float rz0Cos, rz0Sin, rz1Cos, rz1Sin, rxCos, rxSin;
  float t0, t1, t3, t4, t5, t7, t8, t9, t11; 
  float r11, r12, r21, r22, r31, r32;

  //float *Pose = &Poses[Idx * 6];
  tx = Poses[Idx * 6];
  ty = Poses[Idx * 6+1];
  tz = Poses[Idx * 6+2];
  rx = Poses[Idx * 6+3];
  rz0 = Poses[Idx * 6+4];
  rz1 = Poses[Idx * 6+5];
  
  rz0Cos = cosf(rz0); rz0Sin = sinf(rz0);
  rz1Cos = cosf(rz1); rz1Sin = sinf(rz1);
  rxCos = cosf(rx); rxSin = sinf(rx);

	//  z coordinate is y cross x   so add minus
	r11 =  rz0Cos * rz1Cos - rz0Sin * rxCos * rz1Sin;
	r12 = -rz0Cos * rz1Sin - rz0Sin * rxCos * rz1Cos;
	r21 =  rz0Sin * rz1Cos + rz0Cos * rxCos * rz1Sin;
	r22 = -rz0Sin * rz1Sin + rz0Cos * rxCos * rz1Cos;
	r31 =  rxSin * rz1Sin;
	r32 =  rxSin * rz1Cos;
	
	// final transfomration
	t0 = Sf.x*r11 + P.x*r31;
	t1 = Sf.x*r12 + P.x*r32;
	t3 = Sf.x*tx  + P.x*tz;
	t4 = Sf.y*r21 + (P.y-1)*r31;
	t5 = Sf.y*r22 + (P.y-1)*r32;
	t7 = Sf.y*ty  + (P.y-1)*tz;
	t8 = r31;
	t9 = r32;
	t11 = tz;
  
  // reject transformations make marker out of boundary
	float c1x = (t0*(-normDim.x) + t1*(-normDim.y) + t3 ) / 
               (t8*(-normDim.x) + t9*(-normDim.y) + t11);
	float c1y = (t4*(-normDim.x) + t5*(-normDim.y) + t7) /
               (t8*(-normDim.x) + t9*(-normDim.y) + t11 );
	float c2x = (t0*(+normDim.x) + t1*(-normDim.y) + t3) /
               (t8*(+normDim.x) + t9*(-normDim.y) + t11);
	float c2y = (t4*(+normDim.x) + t5*(-normDim.y) + t7) /
               (t8*(+normDim.x) + t9*(-normDim.y) + t11);
	float c3x = (t0*(+normDim.x) + t1*(+normDim.y) + t3) /
               (t8*(+normDim.x) + t9*(+normDim.y) + t11);
	float c3y = (t4*(+normDim.x) + t5*(+normDim.y) + t7) /
               (t8*(+normDim.x) + t9*(+normDim.y) + t11);
	float c4x = (t0*(-normDim.x) + t1*(+normDim.y) + t3) /
               (t8*(-normDim.x) + t9*(+normDim.y) + t11);
	float c4y = (t4*(-normDim.x) + t5*(+normDim.y) + t7) /
               (t8*(-normDim.x) + t9*(+normDim.y) + t11);
	bool isOutofBound = (c1x < 0) | (c1x >= imgDim.x) | (c1y < 0) | (c1y >= imgDim.y) |
		(c2x < 0) | (c2x >= imgDim.x) | (c2y < 0) | (c2y >= imgDim.y) |
		(c3x < 0) | (c3x >= imgDim.x) | (c3y < 0) | (c3y >= imgDim.y) |
		(c4x < 0) | (c4x >= imgDim.x) | (c4y < 0) | (c4y >= imgDim.y);
	if (isOutofBound) {
		Eas[Idx] = INT_MAX;
		return;
	}
  
  // calculate Ea
  float score = 0.0;
  float z;
  float Y, Cr, Cb;
  float u, v;
  for (int i = 0; i < numPoints; i++) {
    // calculate coordinate on camera image
	  z = t8*const_Mcoor[i].x + t9*const_Mcoor[i].y + t11;
	  u = (t0*const_Mcoor[i].x + t1*const_Mcoor[i].y + t3) / z;
	  v = (t4*const_Mcoor[i].x + t5*const_Mcoor[i].y + t7) / z;
    
    // get value from texture
	  Y = tex2D(tex_imgY, u, v);
	  Cr = tex2D(tex_imgCr, u, v);
	  Cb = tex2D(tex_imgCb, u, v);
    
    // calculate distant
    score += (0.5*abs(Y - const_marker[3*i]) + 0.25*abs(Cr - const_marker[3*i+1]) + 0.25*abs(Cb - const_marker[3*i+2]));
  }
  score /= numPoints;
  Eas[Idx] = score;
}


void calEa(float*Eas, float* Poses, const double& Sxf, const double& Syf, const int& x_w, const int& y_h, 
           const double& marker_w, const double& marker_h, const int &wI, const int& hI, const int& numPoses) {
  
  // copy poses and Eas to device memory
  float *buffer_d;
  if (cudaMalloc(&buffer_d, numPoses * 7 * sizeof(float)) != cudaSuccess) {
	  cerr << "out of global memory!\n";
	  exit(-1);
  }
  cudaMemcpy(&buffer_d[0], Poses, numPoses * 6 * sizeof(float), cudaMemcpyHostToDevice);
  //cudaMemset(&buffer_d[numPoses * 6], 0, numPoses * sizeof(float));//
  float *Poses_d = &buffer_d[0];
  float *Eas_d = &buffer_d[numPoses*6];
  //cudaMemcpy(Eas, Eas_d, numPoses * sizeof(float), cudaMemcpyDeviceToHost);//
  //cout << Eas[1021] << "," << Eas[1022] << "," << Eas[1023] << endl;//

  // CUDA Kernel
  Timer timer;
  timer.Reset(); timer.Start();
  unsigned int BLOCK_NUM = (numPoses - 1) / BLOCK_SIZE + 1;
  cout << cudaGetErrorString(cudaGetLastError()) << endl;
  calEa_kernel << < BLOCK_NUM, BLOCK_SIZE >> > (Poses_d, Eas_d, make_float2(Sxf, Syf), make_int2(x_w, y_h), make_float2(marker_w, marker_h), make_int2(wI, hI), numPoses, 444);
  timer.Pause();
  cout << "Kernel: " << timer.get_count() << " ns." << endl;
  cout << cudaGetErrorString(cudaGetLastError()) << endl;
  // copy and free
  cudaMemcpy(Eas, Eas_d, numPoses * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(buffer_d);
}

int main() {
	
  // read image
  Mat img = imread("img.png");
  img.convertTo(img, CV_32FC3, 1/255.0);
  cvtColor(img, img, CV_BGR2YCrCb);
  Mat marker = imread("marker.png");
  marker.convertTo(marker, CV_32FC3, 1/255.0);
  cvtColor(marker, marker, CV_BGR2YCrCb);
  
  // read poses
  const int numPoses = 5166368;
  float *Poses = new float[numPoses * 6];
  float *Eas = new float[numPoses];
  ifstream inFile("poses.txt");
  if (!inFile)
	  return 0;
  for (int i = 0; i < numPoses * 6; i++) {
	 float tmp;
	  inFile >> tmp;
	  Poses[i] = tmp;
  }
  
  //float *Poses = new float[1024 * 6];
  //float *Eas = new float[1024];
  //for (int i = 0; i < 1023; i++) {
	//Poses[6 * i] = -0.64319;
	//Poses[6 * i + 1] = -0.38446;
  //  Poses[6*i+2] = 3;
  //  Poses[6*i+3] = 0;
	//Poses[6 * i + 4] = -3.1416;
	//Poses[6 * i + 5] = -3.1416;
  //}
  //Poses[6138] = -0.1488;
  //Poses[6139] = 0.0855;
  //Poses[6140] = 4.6846;
  //Poses[6141] = 0.1890;
  //Poses[6142] = 0.0398;
  //Poses[6143] = -0.6661;
  
  // get random points
  float marker_w = 0.5 * marker.cols / marker.rows;
  float marker_h = 0.5;

  
  float2 *coor2 = new float2[444];
  float *value = new float[444*3];
  for (int i = 0; i < 444; i++) {
	float x = rand() % marker.cols;
	float y = rand() % marker.rows;
    Vec3f YCrCb = marker.at<Vec3f>(y, x);
    value[3 * i] = YCrCb[0];
    value[3 * i + 1] = YCrCb[1];
    value[3 * i + 2] = YCrCb[2];
	coor2[i].x = (2 * x - marker.cols) / marker.cols * marker_w;
	coor2[i].y = -(2 * y - marker.rows) / marker.rows * marker_h;
  }
  
  
  Mat imgY(img.rows, img.cols, CV_32F);
  Mat imgCr(img.rows, img.cols, CV_32F);
  Mat imgCb(img.rows, img.cols, CV_32F);
  for (int j = 0; j < img.rows; j++) {
	  float* img_j = img.ptr<float>(j);
	  float* imgY_j = imgY.ptr<float>(j);
	  float* imgCr_j = imgCr.ptr<float>(j);
	  float* imgCb_j = imgCb.ptr<float>(j);
	  for (int i = 0; i < img.cols; i++) {
		  imgY_j[i] = img_j[3*i];
		  imgCr_j[i] = img_j[3*i+1];
		  imgCb_j[i] = img_j[3*i+2];
	  }
  }
  Timer timer;
  timer.Reset(); timer.Start();
  cudaMemcpyToSymbol(const_Mcoor, coor2, sizeof(float2)* 444);
  cudaMemcpyToSymbol(const_marker, value, sizeof(float)* 444 * 3);

  // set texture parameters
  tex_imgY.addressMode[0] = cudaAddressModeBorder;
  tex_imgY.addressMode[1] = cudaAddressModeBorder;
  tex_imgY.filterMode = cudaFilterModeLinear;
  tex_imgY.normalized = false;
  tex_imgCr.addressMode[0] = cudaAddressModeBorder;
  tex_imgCr.addressMode[1] = cudaAddressModeBorder;
  tex_imgCr.filterMode = cudaFilterModeLinear;
  tex_imgCr.normalized = false;
  tex_imgCb.addressMode[0] = cudaAddressModeBorder;
  tex_imgCb.addressMode[1] = cudaAddressModeBorder;
  tex_imgCb.filterMode = cudaFilterModeLinear;
  tex_imgCb.normalized = false;
  
  // copy to texture memory
  cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
  cudaArray *imgYArray, *imgCrArray, *imgCbArray;
  cudaMallocArray(&imgYArray, &desc, imgY.cols, imgY.rows);
  cudaMemcpyToArray(imgYArray, 0, 0, imgY.data, sizeof(float)*imgY.cols*imgY.rows, cudaMemcpyHostToDevice);
  cudaBindTextureToArray(tex_imgY, imgYArray, desc);

  cudaMallocArray(&imgCrArray, &desc, imgCr.cols, imgCr.rows);
  cudaMemcpyToArray(imgCrArray, 0, 0, imgCr.data, sizeof(float)*imgCr.cols*imgCr.rows, cudaMemcpyHostToDevice);
  cudaBindTextureToArray(tex_imgCr, imgCrArray, desc);

  cudaMallocArray(&imgCbArray, &desc, imgCb.cols, imgCb.rows);
  cudaMemcpyToArray(imgCbArray, 0, 0, imgCb.data, sizeof(float)*imgCb.cols*imgCb.rows, cudaMemcpyHostToDevice);
  cudaBindTextureToArray(tex_imgCb, imgCbArray, desc);
  
  calEa(Eas, Poses, 1000, -1000, 400, 300, marker_w, marker_h, img.cols, img.rows, numPoses);
  timer.Pause();
  cout << "Overall: " << timer.get_count() << " ns." << endl;
  ofstream outFile("EasCuda.txt");
  if (!outFile)
	  return 0;
  for (int i = 0; i < numPoses; i++) {
	  outFile << Eas[i] << endl;
  }
  //inFile.close();
  outFile.close();
  delete[] Poses;
  delete[] coor2;
  delete[] value;
  delete[] Eas;
  cudaUnbindTexture(tex_imgY);
  cudaUnbindTexture(tex_imgCr);
  cudaUnbindTexture(tex_imgCb);
  cudaFreeArray(imgYArray);
  cudaFreeArray(imgCrArray);
  cudaFreeArray(imgCbArray);
  cout << cudaGetErrorString(cudaGetLastError()) << endl;
  return 0;
}
















