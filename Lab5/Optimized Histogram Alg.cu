#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <wb.h>




#define NUM_BINS 4096

__global__ void PrivatizedHistogram(unsigned int *deviceInput, unsigned int *deviceBins, int inputLength){
	__shared__ int shared_bin[NUM_BINS];

	int i = threadIdx.x;
	while (i < NUM_BINS){
		shared_bin[i] = 0;
		i += blockDim.x;
	}
	__syncthreads();
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < inputLength; i += blockDim.x*gridDim.x){
		atomicAdd(&shared_bin[deviceInput[i]], 1);
	}
	__syncthreads();
	for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x){
		atomicAdd(&deviceBins[i], shared_bin[i]);
	}

}

__global__ void nonPrivatizedHistogram(unsigned int *deviceInput, unsigned int *deviceBins, int inputLength){
	unsigned int i;
	unsigned int offset;

	i = blockIdx.x * blockDim.x + threadIdx.x;
	offset = blockDim.x * gridDim.x;
	while (i < inputLength){
		atomicAdd(&(deviceBins[deviceInput[i]]), 1);
		i += offset;
	}
	__syncthreads();

}

__global__ void modify(unsigned int *deviceBins){
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < NUM_BINS){
		if (deviceBins[i]>127){
			deviceBins[i] = 127;
		}
	}
}


#define CUDA_CHECK(ans)                                                   \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
	bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
			file, line);
		if (abort)
			exit(code);
	}
}

int main(int argc, char *argv[]) {
	wbArg_t args;
	int inputLength;
	unsigned int *hostInput;
	unsigned int *hostBins;
	unsigned int *deviceInput;
	unsigned int *deviceBins;

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 0),
		&inputLength, "Integer");
	hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The input length is ", inputLength);
	wbLog(TRACE, "The number of bins is ", NUM_BINS);

	wbTime_start(GPU, "Allocating GPU memory.");
	// TODO: Allocate GPU memory here
	cudaMalloc((void **)&deviceInput, inputLength*sizeof(unsigned int));
	cudaMalloc((void **)&deviceBins, NUM_BINS*sizeof(unsigned int));


	CUDA_CHECK(cudaDeviceSynchronize());
	wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(GPU, "Copying input memory to the GPU.");
	// TODO: Copy memory to the GPU here
	cudaMemcpy(deviceInput, hostInput, inputLength*sizeof(unsigned int), cudaMemcpyHostToDevice);
	//cudaMemcpy(deviceBins, hostBins, inputLength*sizeof(unsigned int), cudaMemcpyHostToDevice);


	CUDA_CHECK(cudaDeviceSynchronize());
	wbTime_stop(GPU, "Copying input memory to the GPU.");

	dim3 grid(ceil(NUM_BINS / 256.0), 1);
	dim3 block(256, 1, 1);

	// Launch kernel
	// ----------------------------------------------------------

	//printf("======================BEFORE KERNEL\n");
	//PrivatizedHistogram << <grid, block, NUM_BINS*sizeof(float)>> > (deviceInput, deviceBins, inputLength);
	PrivatizedHistogram << <grid, block>> > (deviceInput, deviceBins, inputLength);


	//nonPrivatizedHistogram << <ceil(inputLength / 256.0), 256 >> > (deviceInput, deviceBins, inputLength);

	modify << <ceil(NUM_BINS / 256.0), 256 >> >(deviceBins);
	//printf("======================AFTER KERNEL\n");
	wbLog(TRACE, "Launching kernel");
	wbTime_start(Compute, "Performing CUDA computation");

	// TODO: Perform kernel computation here

	// You should call the following lines after you call the kernel.
	// CUDA_CHECK(cudaGetLastError());
	// CUDA_CHECK(cudaDeviceSynchronize());



	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output memory to the CPU");
	// TODO: Copy the GPU memory back to the CPU here
	cudaMemcpy(hostBins, deviceBins, NUM_BINS*sizeof(unsigned int), cudaMemcpyDeviceToHost);

	CUDA_CHECK(cudaDeviceSynchronize());
	wbTime_stop(Copy, "Copying output memory to the CPU");

	wbTime_start(GPU, "Freeing GPU Memory");
	// TODO: Free the GPU memory here
	cudaFree(deviceInput);
	cudaFree(deviceBins);



	wbTime_stop(GPU, "Freeing GPU Memory");

	// Verify correctness
	// -----------------------------------------------------
	wbSolution(args, hostBins, NUM_BINS);

	free(hostBins);
	free(hostInput);

#if LAB_DEBUG
	system("pause");
#endif

	return 0;
}
