#pragma once

#include "dependencies.h"

__global__ void k_1D_extract_histogram(unsigned char* input, int rows, int cols);
__global__ void k_1D_extract_histogram_shared_mem(unsigned char* input, int rows, int cols);

__global__ void k_1D_normalize_cdf_equalization(int pixels);
__global__ void k_1D_normalize_cdf_equalization_shared_mem(int pixels);

__global__ void k_1D_histogram_equalization(unsigned char* input, int pixels);
__global__ void k_1D_histogram_equalization_shared_mem(unsigned char* input, int pixels);

__global__ void k_3D_extract_histogram(unsigned char* input, int pixels);
__global__ void k_3D_extract_histogram_shared_mem(unsigned char* input, int pixels);

__global__ void k_3D_normalize_cdf_equalization(int pixels);
__global__ void k_3D_normalize_cdf_equalization_shared_mem(int pixels);

__global__ void k_3D_histogram_equalization(unsigned char* input, int rows, int cols) ;
__global__ void k_3D_histogram_equalization_shared_mem(unsigned char* input, int rows, int cols);


float histogram_equalization_gpu_1D(cv::Mat inputImg, cv::Mat* outputImg, bool sm);
float histogram_equalization_gpu_3D(cv::Mat input_img, cv::Mat* output_img, bool sm);

float histogram_equalization_cpu_1D(cv::Mat inputImg, cv::Mat* outputImg);
float histogram_equalization_cpu_3D(cv::Mat input_img, cv::Mat* output_img);

float histogram_equalization_cpu_parallel_1D(cv::Mat inputImg, cv::Mat* outputImg);
float histogram_equalization_cpu_parallel_3D(cv::Mat inputImg, cv::Mat* outputImg);

float histogram_equalization_cpu_parallel_openMP_1D(cv::Mat inputImg, cv::Mat* outputImg);
float histogram_equalization_cpu_parallel_openMP_3D(cv::Mat inputImg, cv::Mat* outputImg);
