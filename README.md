# APEcuda

## CalEa
(texture always hit: 0.05s)
ver1. initial commit, 5s

ver2. float4 const and tex mem, 3.1s

ver3. branch to minmax, 3.0 s 

ver3-1. filterModePoint, 1.5 s

ver4. float4 & float2 pose(transpose pose too), 0.8 s

## getPoses (matlab, 0.4s)
ver1. initial commit, 0.04s

## expandPoses (matlab, 0.8s)
ver1. initial commit, 0.02s

## ranPixels
ver1. initial commit (using device_vector)

## preCal
ver1. initial commit (wx wy 8)
ver2. for opencv2.4
ver3. preallocate gpumat(dont improve)

## creatSet
ver1. initial commit (tz rx nested)

## C2Festimate (matlab, 40 s)
ver1. initial commit (1.5 s)

## APE (matlab, np 5r 60 s, p 5r 60 s)
ver1. initial commit (np 6r 2.7 s, p 7r 5.1 s)
      preCal 0.4 s
      using float4 tmp[SAMPLE_NUM] for p