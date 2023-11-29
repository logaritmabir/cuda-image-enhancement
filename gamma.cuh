#pragma once

#include "dependencies.h"

__global__ void k_3D_gamma_correction_shared_mem(unsigned char* input, int rows, int cols);
__global__ void k_3D_gamma_correction(unsigned char* input, int rows, int cols);
__global__ void k_init_LUT(float gamma);

float gamma_correction_cpu_3D(cv::Mat inputImg, cv::Mat* outputImg, float gamma);
float gamma_correction_cpu_parallel_3D(cv::Mat inputImg, cv::Mat* outputImg, float gamma);
float gamma_correction_cpu_openMP_3D(cv::Mat inputImg, cv::Mat* outputImg, float gamma);

float gamma_correction_gpu_3D(cv::Mat inputImg, cv::Mat* outputImg, float gamma, bool sm);