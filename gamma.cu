#include "gamma.cuh"

__device__ unsigned char LUT_device[256];
__constant__ unsigned char LUT_constant[256];


__global__ void k_init_LUT(float gamma) {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x);

	LUT_device[threadId] = static_cast<unsigned char>(pow(threadId / 255.0f, gamma) * 255);
}

__global__ void k_3D_gamma_correction(unsigned char* input, int rows, int cols) {
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (thread_id >= rows * cols) {
		return;
	}
	__syncthreads();
	input[thread_id] = LUT_device[input[thread_id]];
}


__global__ void k_3D_gamma_correction_shared_mem(unsigned char* input, int rows, int cols) {
	__shared__ unsigned char cache_LUT[256];
		
	int thread_id_in_block = threadIdx.x;
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (thread_id_in_block < 256) {
		cache_LUT[thread_id_in_block] = LUT_device[thread_id_in_block];
	}

	if (thread_id >= rows * cols) {
		return;
	}
	__syncthreads();

	input[thread_id] = cache_LUT[input[thread_id]];
}

float gamma_correction_gpu_3D(cv::Mat input_img, cv::Mat* output_img, float gamma, bool sm) {
	unsigned char* gpu_input = NULL;

	unsigned int cols = input_img.cols * 3;
	unsigned int rows = input_img.rows;
	unsigned long int size = cols * rows * sizeof(unsigned char);

	unsigned char* input = input_img.data;
	unsigned char* output = output_img->data;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	CHECK_CUDA_ERROR(cudaHostAlloc(&input, size, cudaHostAllocDefault));
	memcpy(input, input_img.data, size);

	CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_input, size));
	CHECK_CUDA_ERROR(cudaMemcpy(gpu_input, input, size, cudaMemcpyHostToDevice));

	dim3 block(1024);
	dim3 grid((size + block.x - 1) / block.x);

	if(sm){
		k_init_LUT << <8, 32 >> > (gamma);
		k_3D_gamma_correction_shared_mem << <grid, block >> > (gpu_input, rows, cols);
	}else{
		k_init_LUT << <8, 32 >> > (gamma);
		k_3D_gamma_correction << <grid, block >> > (gpu_input, rows, cols);
	}

	CHECK_CUDA_ERROR(cudaMemcpy(output, gpu_input, size, cudaMemcpyDeviceToHost));

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float gpuElapsedTime = 0;
	cudaEventElapsedTime(&gpuElapsedTime, start, stop);

	cudaFree(gpu_input);
	cudaFreeHost(input);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaDeviceReset();

	return gpuElapsedTime;
}

float gamma_correction_cpu_3D(cv::Mat inputImg, cv::Mat* outputImg, float gamma) {
	unsigned char* input = inputImg.data;
	unsigned char* output = outputImg->data;

	unsigned int rows = inputImg.rows;
	unsigned int cols = inputImg.cols;
	unsigned int pixels = rows * cols * 3;

	auto start = std::chrono::steady_clock::now();

	unsigned char LUT[256] = { 0 };
	for (int i = 0; i < 256; i++) {
		LUT[i] = static_cast<unsigned char>(pow(i / 255.0f, gamma) * 255);
	}
	for (int i = 0; i < pixels; i++) {
		output[i] = LUT[input[i]];
	}

	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start) / 1000.0f;
	return elapsed.count();
}
float gamma_correction_cpu_parallel_3D(cv::Mat inputImg, cv::Mat* outputImg, float gamma) {
	unsigned char* input = inputImg.data;
	unsigned char* output = outputImg->data;

	unsigned int rows = inputImg.rows;
	unsigned int cols = inputImg.cols * 3;

	auto start = std::chrono::steady_clock::now();

	std::vector <std::thread> threads; 
	const int MAX_THREAD_SUPPORT = std::thread::hardware_concurrency();

	int stride = rows / MAX_THREAD_SUPPORT;

	unsigned char LUT[256] = { 0 };

	for (int i = 0; i < 256; i++) {
		LUT[i] = static_cast<uchar>(pow(i / 255.0f, gamma) * 255);
	}

	for (int i = 0; i < MAX_THREAD_SUPPORT; i++) {
		threads.push_back(std::thread([&,i]() {
			int range_start = stride * i;
			int range_end = (i == MAX_THREAD_SUPPORT - 1) ? rows : stride * (i + 1);

			for (int x = range_start; x < range_end; x++) {
				for (int y = 0; y < cols; y++) {
					int index = x * cols + y;
					output[index] = LUT[input[index]];
				}
			}
			}));
	}
	for (std::thread& thread : threads) {
		thread.join();
	}

	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start) / 1000.0f;
	return elapsed.count();
}

float gamma_correction_cpu_openMP_3D(cv::Mat inputImg, cv::Mat* outputImg, float gamma) {
	unsigned char* input = inputImg.data;
	unsigned char* output = outputImg->data;

	unsigned int rows = inputImg.rows;
	unsigned int cols = inputImg.cols * 3;

	auto start = std::chrono::steady_clock::now();

	const int MAX_THREAD_SUPPORT = std::thread::hardware_concurrency();

	unsigned char LUT[256] = { 0 };

	#pragma omp parallel for num_threads(MAX_THREAD_SUPPORT)
	for (int i = 0; i < 256; i++) {
		LUT[i] = static_cast<unsigned char>(pow(i / 255.0f, gamma) * 255);
	}

	#pragma omp parallel for num_threads(MAX_THREAD_SUPPORT)
	for (int x = 0; x < rows; x++) {
		for (int y = 0; y < cols; y++) {
			int index = x * cols + y;
			output[index] = LUT[input[index]];
		}
	}

	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start) / 1000.0f;
	return elapsed.count();
}