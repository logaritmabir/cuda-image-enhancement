#ifndef __GPU__

#define __GPU__
#include "gaussian.cuh"
#include "sobel.cuh"
#include "gamma.cuh"
#include "histogram.cuh"

#endif

#include <string>
#include <cmath>
#include <chrono>
#include <math.h>

int main()
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT); /*supressing annoying opencv infos in cmdline*/

	std::string img_path = "C:/Users/ben/Desktop/cuda-img-enhancement/images/4096.png";
	cv::Mat color_img = cv::imread(img_path, cv::IMREAD_COLOR);

	cv::Mat gray_img;
	cv::cvtColor(color_img, gray_img, cv::COLOR_BGR2GRAY);

	cv::Mat color_cpu_img = color_img.clone();
	cv::Mat gray_cpu_img = gray_img.clone();

	cv::Mat color_cpu_parallel_img = color_img.clone();
	cv::Mat gray_cpu_parallel_img = gray_img.clone();

	cv::Mat color_gpu_img = color_img.clone();
	cv::Mat gray_gpu_img = gray_img.clone();
	
	float gamma = 0.3f;

	float cpu_3d_elapsed = 0, cpu_1d_elapsed = 0, gpu_1d_elapsed = 0, gpu_1d_sm_elapsed = 0, gpu_3d_elapsed = 0, gpu_3d_sm_elapsed = 0;
	float cpu_1d_parallel_elapsed = 0, cpu_3d_parallel_elapsed = 0;
	int iter = 1;

	for(int i = 0 ; i < iter ; i++){
		//cpu_1d_elapsed += gaussian_filter_cpu_1D(gray_img, &gray_cpu_img);
		//cpu_3d_elapsed += gaussian_filter_cpu_3D(color_img, &color_cpu_img);

		//cpu_1d_parallel_elapsed += gaussian_filter_cpu_openMP_1D(gray_img, &gray_cpu_parallel_img);
		//cpu_3d_parallel_elapsed += gaussian_filter_cpu_openMP_3D(color_img, &color_cpu_parallel_img);

		//gpu_1d_sm_elapsed += gaussian_filter_gpu_1D(gray_img, &gray_gpu_img, true);
		//gpu_1d_elapsed += gaussian_filter_gpu_1D(gray_img, &gray_gpu_img, false);
		
		gpu_3d_sm_elapsed += gaussian_filter_gpu_3D(color_img, &color_gpu_img, true);
		gpu_3d_elapsed += gaussian_filter_gpu_3D(color_img, &color_gpu_img, false);
		
	}

	cpu_3d_elapsed /= iter;
	cpu_1d_elapsed /= iter;
	gpu_1d_elapsed /= iter;
	gpu_1d_sm_elapsed /= iter;
	gpu_3d_elapsed /= iter;
	gpu_3d_sm_elapsed /= iter;
	cpu_1d_parallel_elapsed /= iter;
	cpu_3d_parallel_elapsed /= iter;

	std::cout << "cpu_1d_elapsed: " << cpu_1d_elapsed << "\n";
	std::cout << "cpu_1d_parallel_elapsed: " << cpu_1d_parallel_elapsed << "\n";
	std::cout << "gpu_1d_elapsed: " << gpu_1d_elapsed << "\n";
	std::cout << "gpu_1d_sm_elapsed: " << gpu_1d_sm_elapsed << "\n \n";

	std::cout << "cpu_3d_elapsed: " << cpu_3d_elapsed << "\n";
	std::cout << "cpu_3d_parallel_elapsed: " << cpu_3d_parallel_elapsed << "\n";
	std::cout << "gpu_3d_elapsed: " << gpu_3d_elapsed << "\n";
	std::cout << "gpu_3d_sm_elapsed: " << gpu_3d_sm_elapsed << "\n";

	std::cout << "Acceleration of 1d image with multithread: " << cpu_1d_elapsed / cpu_1d_parallel_elapsed << "\n";
	std::cout << "Acceleration of 1d image with GPU: " << cpu_1d_elapsed / gpu_1d_elapsed << "\n";
	std::cout << "Acceleration of 1d image with GPU-SM: " << cpu_1d_elapsed / gpu_1d_sm_elapsed << "\n";

	std::cout << "Acceleration of 3d image with multithread: " << cpu_3d_elapsed / cpu_3d_parallel_elapsed << "\n";
	std::cout << "Acceleration of 3d image with GPU: " << cpu_3d_elapsed / gpu_3d_elapsed << "\n";
	std::cout << "Acceleration of 3d image with GPU-SM: " << cpu_3d_elapsed / gpu_3d_sm_elapsed << "\n";

	std::cout << "Acceleration of shared mem in 1d image: " << gpu_1d_elapsed / gpu_1d_sm_elapsed << "\n";
	std::cout << "Acceleration of shared mem in 3d image: " << gpu_3d_elapsed / gpu_3d_sm_elapsed << "\n";

	cv::imshow("img", color_img);

	cv::imshow("img-gray", gray_img);

	cv::imshow("cpu-gray", gray_cpu_img);
	cv::imshow("cpu-color", color_cpu_img);

	cv::imshow("cpu-gray-multithreaded", gray_cpu_parallel_img);
	cv::imshow("cpu-color-multithreaded", color_cpu_parallel_img);
	
	cv::imshow("gpu-gray", gray_gpu_img);
	cv::imshow("gpu-color", color_gpu_img);

	cv::waitKey(0);
}

/*"Include Directories" C:\opencv\build\include*/