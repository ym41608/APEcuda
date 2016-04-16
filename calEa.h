#ifndef CALEA_H
#define CALEA_H

#include <thrust/device_vector.h>
#include "parameter.h"
#include "ranPixels.h"

__global__ void calEa_NP_kernel(float4 *Poses4, float2 *Poses2, float *Eas, const float2 Sf, const int2 P, const float2 normDim, const int2 imgDim,
  const int numPoses) {
  const int tIdx = threadIdx.x;
  const int Idx = blockIdx.x * 256 + tIdx;

  if (Idx >= numPoses)
    return;

  // calculate transformation
  float tx, ty, tz, rx, rz0, rz1;
  float rz0Cos, rz0Sin, rz1Cos, rz1Sin, rxCos, rxSin;
  float t0, t1, t3, t4, t5, t7, t8, t9, t11;
  float r11, r12, r21, r22, r31, r32;

  // get pose parameter
  tx = Poses4[Idx].x;
  ty = Poses4[Idx].y;
  tz = Poses4[Idx].z;
  rx = Poses4[Idx].w;
  rz0 = Poses2[Idx].x;
  rz1 = Poses2[Idx].y;

  rz0Cos = cosf(rz0); rz0Sin = sinf(rz0);
  rz1Cos = cosf(rz1); rz1Sin = sinf(rz1);
  rxCos = cosf(rx); rxSin = sinf(rx);

  //  z coordinate is y cross x   so add minus
  r11 = rz0Cos * rz1Cos - rz0Sin * rxCos * rz1Sin;
  r12 = -rz0Cos * rz1Sin - rz0Sin * rxCos * rz1Cos;
  r21 = rz0Sin * rz1Cos + rz0Cos * rxCos * rz1Sin;
  r22 = -rz0Sin * rz1Sin + rz0Cos * rxCos * rz1Cos;
  r31 = rxSin * rz1Sin;
  r32 = rxSin * rz1Cos;

  // final transfomration
  t0 = Sf.x*r11 + P.x*r31;
  t1 = Sf.x*r12 + P.x*r32;
  t3 = Sf.x*tx + P.x*tz;
  t4 = (-Sf.y)*r21 + (P.y - 1)*r31;
  t5 = (-Sf.y)*r22 + (P.y - 1)*r32;
  t7 = (-Sf.y)*ty + (P.y - 1)*tz;
  t8 = r31;
  t9 = r32;
  t11 = tz;

  // reject transformations make marker out of boundary
  float invc1z = 1 / (t8*(-normDim.x) + t9*(-normDim.y) + t11);
  float c1x = (t0*(-normDim.x) + t1*(-normDim.y) + t3) * invc1z;
  float c1y = (t4*(-normDim.x) + t5*(-normDim.y) + t7) * invc1z;
  float invc2z = 1 / (t8*(+normDim.x) + t9*(-normDim.y) + t11);
  float c2x = (t0*(+normDim.x) + t1*(-normDim.y) + t3) * invc2z;
  float c2y = (t4*(+normDim.x) + t5*(-normDim.y) + t7) * invc2z;
  float invc3z = 1 / (t8*(+normDim.x) + t9*(+normDim.y) + t11);
  float c3x = (t0*(+normDim.x) + t1*(+normDim.y) + t3) * invc3z;
  float c3y = (t4*(+normDim.x) + t5*(+normDim.y) + t7) * invc3z;
  float invc4z = 1 / (t8*(-normDim.x) + t9*(+normDim.y) + t11);
  float c4x = (t0*(-normDim.x) + t1*(+normDim.y) + t3) * invc4z;
  float c4y = (t4*(-normDim.x) + t5*(+normDim.y) + t7) * invc4z;
  float minx = min(c1x, min(c2x, min(c3x, c4x)));
  float maxx = max(c1x, max(c2x, max(c3x, c4x)));
  float miny = min(c1y, min(c2y, min(c3y, c4y)));
  float maxy = max(c1y, max(c2y, max(c3y, c4y)));
  if ((minx < 0) | (maxx >= imgDim.x) | (miny < 0) | (maxy >= imgDim.y)) {
    Eas[Idx] = FLT_MAX;
    return;
  }

  // calculate Ea
  float score = 0.0;
  float invz;
  float4 YCrCb_tex, YCrCb_const;
  float u, v;
  for (int i = 0; i < SAMPLE_NUM; i++) {
    
    // calculate coordinate on camera image
    invz = 1 / (t8*const_Mcoor[i].x + t9*const_Mcoor[i].y + t11);
    u = (t0*const_Mcoor[i].x + t1*const_Mcoor[i].y + t3) * invz;
    v = (t4*const_Mcoor[i].x + t5*const_Mcoor[i].y + t7) * invz;

    // get value from constmem
    YCrCb_const = const_marker[i];

    // get value from texture
    YCrCb_tex = tex2D(tex_imgYCrCb, u, v);

    // calculate distant
    score += (2.852 * abs(YCrCb_tex.x - YCrCb_const.x) + abs(YCrCb_tex.y - YCrCb_const.y) + 1.264 * abs(YCrCb_tex.z - YCrCb_const.z));
  }
  Eas[Idx] = score / (SAMPLE_NUM * 5.116);
};

__global__ void calEa_P_kernel(float4 *Poses4, float2 *Poses2, float *Eas, const float2 Sf, const int2 P, const float2 normDim, const int2 imgDim, 
  const int numPoses) {
  const int tIdx = threadIdx.x;
  const int Idx = blockIdx.x * BLOCK_SIZE + tIdx;

  if (Idx >= numPoses)
    return;

  // calculate transformation
  float tx, ty, tz, rx, rz0, rz1;
  float rz0Cos, rz0Sin, rz1Cos, rz1Sin, rxCos, rxSin;
  float t0, t1, t3, t4, t5, t7, t8, t9, t11;
  float r11, r12, r21, r22, r31, r32;

  // get pose parameter
  tx = Poses4[Idx].x;
  ty = Poses4[Idx].y;
  tz = Poses4[Idx].z;
  rx = Poses4[Idx].w;
  rz0 = Poses2[Idx].x;
  rz1 = Poses2[Idx].y;

  rz0Cos = cosf(rz0); rz0Sin = sinf(rz0);
  rz1Cos = cosf(rz1); rz1Sin = sinf(rz1);
  rxCos = cosf(rx); rxSin = sinf(rx);

  //  z coordinate is y cross x   so add minus
  r11 = rz0Cos * rz1Cos - rz0Sin * rxCos * rz1Sin;
  r12 = -rz0Cos * rz1Sin - rz0Sin * rxCos * rz1Cos;
  r21 = rz0Sin * rz1Cos + rz0Cos * rxCos * rz1Sin;
  r22 = -rz0Sin * rz1Sin + rz0Cos * rxCos * rz1Cos;
  r31 = rxSin * rz1Sin;
  r32 = rxSin * rz1Cos;

  // final transfomration
  t0 = Sf.x*r11 + P.x*r31;
  t1 = Sf.x*r12 + P.x*r32;
  t3 = Sf.x*tx + P.x*tz;
  t4 = (-Sf.y)*r21 + (P.y - 1)*r31;
  t5 = (-Sf.y)*r22 + (P.y - 1)*r32;
  t7 = (-Sf.y)*ty + (P.y - 1)*tz;
  t8 = r31;
  t9 = r32;
  t11 = tz;

  // reject transformations make marker out of boundary
  float invc1z = 1 / (t8*(-normDim.x) + t9*(-normDim.y) + t11);
  float c1x = (t0*(-normDim.x) + t1*(-normDim.y) + t3) * invc1z;
  float c1y = (t4*(-normDim.x) + t5*(-normDim.y) + t7) * invc1z;
  float invc2z = 1 / (t8*(+normDim.x) + t9*(-normDim.y) + t11);
  float c2x = (t0*(+normDim.x) + t1*(-normDim.y) + t3) * invc2z;
  float c2y = (t4*(+normDim.x) + t5*(-normDim.y) + t7) * invc2z;
  float invc3z = 1 / (t8*(+normDim.x) + t9*(+normDim.y) + t11);
  float c3x = (t0*(+normDim.x) + t1*(+normDim.y) + t3) * invc3z;
  float c3y = (t4*(+normDim.x) + t5*(+normDim.y) + t7) * invc3z;
  float invc4z = 1 / (t8*(-normDim.x) + t9*(+normDim.y) + t11);
  float c4x = (t0*(-normDim.x) + t1*(+normDim.y) + t3) * invc4z;
  float c4y = (t4*(-normDim.x) + t5*(+normDim.y) + t7) * invc4z;
  float minx = min(c1x, min(c2x, min(c3x, c4x)));
  float maxx = max(c1x, max(c2x, max(c3x, c4x)));
  float miny = min(c1y, min(c2y, min(c3y, c4y)));
  float maxy = max(c1y, max(c2y, max(c3y, c4y)));
  if ((minx < 0) | (maxx >= imgDim.x) | (miny < 0) | (maxy >= imgDim.y)) {
    Eas[Idx] = FLT_MAX;
    return;
  }

  // calculate Ea
  float score = 0.0;
  float invz;
  float4 YCrCb_tex, YCrCb_const;
  float u, v;
  
  // parameters for normalization
  float sumXi = 0; float sumYi = 0;
  float sumXiSqrd = 0; float sumYiSqrd = 0;
  float Xi, Yi;
  float4 tmpp[SAMPLE_NUM];
  for (int i = 0; i < SAMPLE_NUM; i++) {
    // calculate coordinate on camera image
    invz = 1 / (t8*const_Mcoor[i].x + t9*const_Mcoor[i].y + t11);
    u = (t0*const_Mcoor[i].x + t1*const_Mcoor[i].y + t3) * invz;
    v = (t4*const_Mcoor[i].x + t5*const_Mcoor[i].y + t7) * invz;

    // get value from constmem
    YCrCb_const = const_marker[i];

    // get value from texture
    YCrCb_tex = tex2D(tex_imgYCrCb, u, v);
    
    // accumulation for normalization
    Xi = YCrCb_const.x;
    Yi = YCrCb_tex.x;
    tmpp[i] = YCrCb_tex;
    sumXi += Xi;
    sumYi += Yi;
    sumXiSqrd += (Xi*Xi);
    sumYiSqrd += (Yi*Yi);
  }
  
  // normalization parameter
  float sigX = sqrt((sumXiSqrd-(sumXi*sumXi)/SAMPLE_NUM)/SAMPLE_NUM) + 0.0000001;
  float sigY = sqrt((sumYiSqrd-(sumYi*sumYi)/SAMPLE_NUM)/SAMPLE_NUM) + 0.0000001;
  float meanX = sumXi/SAMPLE_NUM;
  float meanY = sumYi/SAMPLE_NUM;
  float sigXoversigY = sigX/sigY;
  float faster = -meanX+sigXoversigY*meanY;
  
  for (int i = 0; i < SAMPLE_NUM; i++) {
    YCrCb_const = const_marker[i];
    YCrCb_tex = tmpp[i];
    score += (2.852*abs(YCrCb_const.x - sigXoversigY*YCrCb_tex.x+faster) + abs(YCrCb_tex.y - YCrCb_const.y) + 1.264*abs(YCrCb_tex.z - YCrCb_const.z));
  }
  Eas[Idx] = score / (SAMPLE_NUM * 5.116);
};

#endif