#ifndef IM_CONV_H
#define IM_CONV_H

#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

using namespace std;
using namespace cv;

Mat im2double(Mat A);

cuda::GpuMat im2double(cuda::GpuMat A);

Mat im2uint8(Mat A);

cuda::GpuMat im2uint8(cuda::GpuMat A);

Mat rgb2ntsc(Mat A);

cuda::GpuMat rgb2ntsc(cuda::GpuMat A);

Mat ntsc2rgb(Mat A);

cuda::GpuMat ntsc2rgb(cuda::GpuMat A);

#endif /* IM_CONV_H */