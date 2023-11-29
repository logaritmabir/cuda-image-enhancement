#pragma once

#include "dependencies.h"

__global__ void k_1D_gaussian_filter(unsigned char* input, int rows, int cols, int mask_dim);
__global__ void k_1D_gaussian_filter_shared_mem(unsigned char* input, int rows, int cols, int mask_dim);

__global__ void k_3D_gaussian_filter(unsigned char *input, int rows, int cols, int mask_dim);
__global__ void k_3D_gaussian_filter_shared_mem(unsigned char *input, int rows, int cols, int mask_dim);

float gaussian_filter_cpu_1D(cv::Mat input_img, cv::Mat* output_img);
float gaussian_filter_cpu_3D(cv::Mat input_img, cv::Mat *output_img);

float gaussian_filter_cpu_parallel_1D(cv::Mat input_img, cv::Mat* output_img);
float gaussian_filter_cpu_parallel_3D(cv::Mat input_img, cv::Mat* output_img);

float gaussian_filter_gpu_1D(cv::Mat input_img, cv::Mat* output_img, bool sm);
float gaussian_filter_gpu_3D(cv::Mat input_img, cv::Mat* output_img, bool sm);





