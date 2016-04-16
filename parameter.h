#ifndef PARAMETER_H
#define PARAMETER_H

static const int BLOCK_SIZE = 256;
static const int SAMPLE_NUM = 444;

// constant
__constant__ float2 const_Mcoor[SAMPLE_NUM];
__constant__ float4 const_marker[SAMPLE_NUM];

// Texture
texture<float4, cudaTextureType2D, cudaReadModeElementType> tex_imgYCrCb;

// class parameter
class parameter {
  public:
    void shrinkNet(const float& factor) {
      s4.x *= factor;
      s4.y *= factor;
      s4.z *= factor;
      s4.w *= factor;
      s2.x *= factor;
      s2.y *= factor;
      delta *= factor;
    };
    
    // about pose net
    float delta;
    float2 Btz;
    float2 Brx;
    float2 Brz;
    float4 s4;
    float2 s2;
    
    // about camera
    float2 Sf;
    int2 P;
    
    // about marker
    int2 mDim;
    float2 markerDim;
    
    // about img
    int2 iDim;
};

#endif