#pragma once

#include "dependencies.h"

__global__ void sobel_filter(unsigned char* input, unsigned char* output, int cols, int rows, int mask_dim);
__global__ void shared_sobel_filter(unsigned char* input, unsigned char* output, int cols, int rows, int mask_dim);

float sobel_filter_gpu(cv::Mat* inputImg, cv::Mat outputImg);
float sobel_filter_cpu(cv::Mat* inputImg, cv::Mat outputImg);
float sobel_filter_cpu_parallel(cv::Mat* inputImg, cv::Mat outputImg);