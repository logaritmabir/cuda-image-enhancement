#include "sobel.cuh"

constexpr int SOBEL_FILTER_SIZE = 3;

__constant__ int filter_x_constant[SOBEL_FILTER_SIZE][SOBEL_FILTER_SIZE] = { 0 };
__constant__ int filter_y_constant[SOBEL_FILTER_SIZE][SOBEL_FILTER_SIZE] = { 0 };

__global__ void sobel_filter(unsigned char* input, unsigned char* output, int cols, int rows, int mask_dim) {
	int ty = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;
	int threadId = tx * cols + ty;

	int offset = mask_dim / 2;
	int gx = 0;
	int gy = 0;

	if (tx > 0 && tx < rows - 1 && ty > 0 && ty < cols - 1) {
		for (int i = 0; i < mask_dim; i++) {
			for (int j = 0; j < mask_dim; j++) {
				gx += filter_x_constant[i][j] * input[(tx - offset + i) * cols + (ty - offset + j)];
				gy += filter_y_constant[i][j] * input[(tx - offset + i) * cols + (ty - offset + j)];
			}
		}
		output[threadId] = static_cast<unsigned char>(sqrtf(static_cast<float>(gx) * static_cast<float>(gx) + static_cast<float>(gy) * static_cast<float>(gy)));
	}
}
__global__ void shared_sobel_filter(unsigned char* input, unsigned char* output, int cols, int rows, int mask_dim) {
	__shared__ unsigned char cache[34][34];
	int tx = blockIdx.y * blockDim.y + threadIdx.y;
	int ty = blockIdx.x * blockDim.x + threadIdx.x;
	int threadId = tx * cols + ty;

	int offset = mask_dim / 2;
	int gx = 0;
	int gy = 0;

	int cy = threadIdx.x + 1;
	int cx = threadIdx.y + 1;

	cache[cx][cy] = input[tx * cols + ty]; /* Load cache[1::32][1::32]*/
	if (cx == 1) {/*Load left column*/
		cache[0][cy] = input[tx * cols + ty - 1];
		if (cy == 1) {
			cache[0][0] = input[(tx - 1) * cols + ty - 1];
		}
		if (cy == 32) {
			cache[33][0] = input[(tx + 1) * cols + ty - 1];
		}
	}
	if (cx == 32) {/*Load right column*/
		cache[33][cy] = input[tx * cols + ty + 1];
		if (cy == 32) {
			cache[33][33] = input[(tx + 1) * cols + ty + 1];
		}
		if (cy == 1) {
			cache[0][33] = input[(tx - 1) * cols + ty + 1];
		}
	}
	if (cy == 1) {/*Load top row*/
		cache[cx][0] = input[(tx - 1) * cols + ty];
	}
	if (cy == 32) {/*Load bottom row*/
		cache[cx][33] = input[(tx + 1) * cols + ty];
	}
	__syncthreads();

	if (ty > 0 && ty < cols - 1 && tx > 0 && tx < rows - 1) {
		for (int i = 0; i < mask_dim; i++) {
			for (int j = 0; j < mask_dim; j++) {
				gx += filter_x_constant[i][j] * cache[cx - offset + i][cy - offset + j];
				gy += filter_y_constant[i][j] * cache[cx - offset + i][cy - offset + j];
			}
		}
		output[threadId] = static_cast<unsigned char>(sqrtf(static_cast<float>(gx) * static_cast<float>(gx) + static_cast<float>(gy) * static_cast<float>(gy)));
	}
}

float sobel_filter_gpu(cv::Mat* inputImg, cv::Mat outputImg) {
	unsigned char* input = inputImg->data;
	unsigned char* output = outputImg.data;

	unsigned char* gpu_input = NULL;
	unsigned char* gpu_output = NULL;

	unsigned int cols = inputImg->cols;
	unsigned int rows = inputImg->rows;
	unsigned int pixels = cols * rows;
	unsigned int size = pixels * sizeof(unsigned char);

	const unsigned int mask_dim = 3;
	int filter_x[3][3] = { {-1 , 0 , 1} , {-2 , 0 , 2 } , {-1 , 0 , 1} };
	int filter_y[3][3] = { {-1 , -2 , -1} , {0 , 0 , 0 } , {1 , 2 , 1} };

	cudaEvent_t beginKernel, endKernel, start, stop;
	cudaEventCreate(&beginKernel);
	cudaEventCreate(&endKernel);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	CHECK_CUDA_ERROR(cudaMemcpyToSymbol(filter_x_constant, filter_x, sizeof(int) * SOBEL_FILTER_SIZE * SOBEL_FILTER_SIZE));
	CHECK_CUDA_ERROR(cudaMemcpyToSymbol(filter_y_constant, filter_y, sizeof(int) * SOBEL_FILTER_SIZE * SOBEL_FILTER_SIZE));
	CHECK_CUDA_ERROR(cudaMalloc((unsigned char**)&gpu_input, size));
	CHECK_CUDA_ERROR(cudaMalloc((unsigned char**)&gpu_output, size));
	CHECK_CUDA_ERROR(cudaMemcpy(gpu_input, input, size, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(gpu_output, output, size, cudaMemcpyHostToDevice));

	dim3 block(32, 32);
	dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

	cudaEventRecord(beginKernel);
	shared_sobel_filter << <grid, block >> > (gpu_input, gpu_output, cols, rows, mask_dim);
	cudaEventRecord(endKernel);

	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	CHECK_CUDA_ERROR(cudaMemcpy(output, gpu_output, size, cudaMemcpyDeviceToHost));

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventSynchronize(endKernel);
	float elapsedKernel = 0.0f;
	float elapsedAll = 0.0f;

	cudaEventElapsedTime(&elapsedKernel, beginKernel, endKernel);
	cudaEventElapsedTime(&elapsedAll, start, stop);
	//printf("---- Sobel Filter ----\n");
	//printf("Total elapsed time in GPU (memory transfers are included) : %3.4f ms\n", elapsedAll);
	//printf("Elapsed time in GPU Kernel : %3.4f ms\n", elapsedKernel);
	//printf("---- ---------------------- ----\n");
	cudaFree(gpu_input);
	cudaFree(gpu_output);
	cudaDeviceReset();
	return elapsedAll;
}
float sobel_filter_cpu(cv::Mat* inputImg, cv::Mat outputImg) {
	unsigned int offset = SOBEL_FILTER_SIZE / 2;
	const int rows = inputImg->rows;
	const int cols = inputImg->cols;

	int filter_x[3][3] = { {-1 , 0 , 1} , {-2 , 0 , 2 } , {-1 , 0 , 1} };
	int filter_y[3][3] = { {-1 , -2 , -1} , {0 , 0 , 0 } , {1 , 2 , 1} };

	auto begin = std::chrono::steady_clock::now();

	for (int i = 1; i < rows - 1; i++) {
		for (int j = 1; j < cols - 1; j++) {
			int gx = 0;
			int gy = 0;
			for (int m = 0; m < SOBEL_FILTER_SIZE; m++) {
				for (int n = 0; n < SOBEL_FILTER_SIZE; n++) {
					gx += inputImg->at<uchar>(i + m - offset, j + n - offset) * filter_x[m][n];
					gy += inputImg->at<uchar>(i + m - offset, j + n - offset) * filter_y[m][n];
				}
			}
			outputImg.at<uchar>(i, j) = static_cast<uchar>(sqrt(gx * gx + gy * gy));
		}
	}

	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
	return elapsed.count();
}
float sobel_filter_cpu_parallel(cv::Mat* inputImg, cv::Mat outputImg) {
	unsigned char* input = inputImg->data;
	unsigned char* output = outputImg.data;
	int cols = inputImg->cols;
	int rows = inputImg->rows;
	const unsigned short mask_dim = 3;

	int filter_x[3][3] = { {-1 , 0 , 1} , {-2 , 0 , 2 } , {-1 , 0 , 1} };
	int filter_y[3][3] = { {-1 , -2 , -1} , {0 , 0 , 0 } , {1 , 2 , 1} };

	std::vector <std::thread> threads;
	const int MAX_THREAD_SUPPORT = std::thread::hardware_concurrency();

	int stride = rows / MAX_THREAD_SUPPORT;

	auto begin = std::chrono::steady_clock::now();
	for (int i = 0; i < MAX_THREAD_SUPPORT; i++) {
		threads.push_back(std::thread([&, i]() {
			int range_start = stride * i;
			int range_end = (i == MAX_THREAD_SUPPORT - 1) ? cols : stride * (i + 1);

			for (int r = range_start; r < range_end; r++) { /*row loop*/
				for (int c = 0; c < cols; c++) { /*col loop*/
					if (r > 0 && r < rows - 1 && c > 0 && c < cols - 1) {
						int gx = 0;
						int gy = 0;
						for (int mr = 0; mr < mask_dim; mr++) { /*matrix row*/
							for (int mc = 0; mc < mask_dim; mc++) { /*matrix col*/
								int r_index = r + mr - 1;
								int c_index = c + mc - 1;
								gx += input[r_index * cols + c_index] * filter_x[mr][mc];
								gy += input[r_index * cols + c_index] * filter_y[mr][mc];
							}
						}
						output[r * cols + c] = static_cast<uchar>(sqrt(gx * gx + gy * gy));
					}
				}
			}
			}));
	}
	for (std::thread& th : threads) {
		th.join();
	}
	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
	return elapsed.count();
}