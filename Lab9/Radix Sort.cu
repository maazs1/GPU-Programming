#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <wb.h>

#define BLOCK_SIZE 512 //TODO: You can change this


#define wbCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void scatter(int *g_odata, int *g_idata, int *position, int n, int number) {
	int ID = threadIdx.x + blockDim.x * blockIdx.x;
	if (ID < n){
		if (number == 2 && g_odata!=NULL){
			g_odata[position[ID]] = g_idata[ID];
			__syncthreads();
		}
		else if (number == 1){
			int storageValue = n - 1;
			int x = g_odata[storageValue];
			x+= position[storageValue];
			__syncthreads();
			if (g_odata[ID] == 0) {
				__syncthreads();
				position[ID] = ID - position[ID] + x;
			}
		}
	}
}

__global__ void checkBits(int *output, int *input, int bitNum, int n) {
	int ID = threadIdx.x + blockDim.x * blockIdx.x;
	int value = 0;
	if (ID < n) {
		value = input[ID] & (1 << bitNum); //(1 << N) SET the particular bit at Nth position
		if (value > 0){
			value = 1;
		}
		output[ID] = 1 - value;
		__syncthreads();
	}
}

__global__ void blockadd(int* g_odata, int* g_aux, int n){
	int ID = blockIdx.x*blockDim.x + threadIdx.x;
	if (ID < n){
		g_odata[ID] += g_aux[blockIdx.x];
	}
}

__global__ void scan(int *g_odata, int *g_idata, int *g_aux, int n){
	__shared__ float sharedMEM[2 * BLOCK_SIZE];
	int ty = threadIdx.x;
	int ID = blockIdx.x*blockDim.x + threadIdx.x;

	if (ID < n){
		sharedMEM[ty] = g_idata[ID];
	}
	else{
		sharedMEM[ty] = 0;
	}
	int stride = 1;
	while (stride <= blockDim.x){
		__syncthreads();
		if (threadIdx.x < stride){
			g_odata[0] = NULL;
			g_odata[ID + 1] = sharedMEM[ty];
		}
		else{
			float val = sharedMEM[ty - stride];
			__syncthreads();
			sharedMEM[ty] += val;
		}
		__syncthreads();
		stride *= 2;
	}
	if (ty == BLOCK_SIZE - 1){
		if (g_aux != NULL){
			g_aux[blockIdx.x] = g_odata[ID + 1];
			g_odata[ID + 1] = 0;
			if (ty == 0){
				g_aux[blockIdx.x] = sharedMEM[BLOCK_SIZE - 1];
			}
		}
	}
}

void recursive_scan(int* g_odata, int* g_idata, int len){
	dim3 block(BLOCK_SIZE, 1);
	int numBlocks = (len / BLOCK_SIZE) + 1;
	dim3 grid(numBlocks, 1);
	int* g_aux;

	if (numBlocks == 1){
		g_aux = NULL;
		scan << <grid, block >> >(g_odata, g_idata, g_aux, len);
		cudaDeviceSynchronize();
	}
	else{
		cudaMalloc((void**)&g_aux, (numBlocks*sizeof(int)));
		int *secondValue;
		cudaMalloc((void**)&secondValue, (numBlocks*sizeof(int)));
		scan << <grid, block >> >(g_odata, g_idata, g_aux, len);
		cudaDeviceSynchronize();
		scan << <1, block >> >(secondValue, g_aux, NULL, numBlocks);
		cudaDeviceSynchronize();
		recursive_scan(secondValue, g_aux, numBlocks);
		blockadd << <grid, block >> >(g_odata, secondValue, len);
		cudaDeviceSynchronize();
		cudaFree(g_aux);
		cudaFree(secondValue);
	}
}

void sort(int* deviceInput, int *deviceOutput, int numElements)
{
	int *valueA;
	cudaMalloc((void **)&valueA, numElements*sizeof(int));
	int* adress;

	dim3 blockDim(BLOCK_SIZE);
	int numBlocks = ceil((float)numElements) / BLOCK_SIZE + 1;
	dim3 gridDim(numBlocks);
	int val = 0;
	for (int bits = 0; bits < 16; bits++)
	{
		checkBits << <gridDim, blockDim >> >(deviceOutput, deviceInput, bits, numElements);
		recursive_scan(valueA, deviceOutput, numElements);
		for (int i = 1; i < 3; i++){
			scatter << <gridDim, blockDim >> > (deviceOutput, deviceInput, valueA, numElements, i);
		}
		adress = deviceInput;
		deviceInput = deviceOutput;
		deviceOutput = adress;
	}
}

int main(int argc, char **argv) {
	wbArg_t args;
	int *hostInput;  // The input 1D list
	int *hostOutput; // The output list
	int *deviceInput;
	int *deviceOutput;
	int numElements; // number of elements in the list

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput = (int *)wbImport(wbArg_getInputFile(args, 0), &numElements, "integral_vector");
	cudaHostAlloc(&hostOutput, numElements * sizeof(int), cudaHostAllocDefault);
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The number of input elements in the input is ", numElements);

	wbTime_start(GPU, "Allocating GPU memory.");
	wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(int)));
	wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(int)));
	wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(GPU, "Clearing output memory.");
	wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(int)));
	wbTime_stop(GPU, "Clearing output memory.");

	wbTime_start(GPU, "Copying input memory to the GPU.");
	wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(int),
		cudaMemcpyHostToDevice));
	wbTime_stop(GPU, "Copying input memory to the GPU.");

	wbTime_start(Compute, "Performing CUDA computation");
	sort(deviceInput, deviceOutput, numElements);
	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output memory to the CPU");
	wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
		cudaMemcpyDeviceToHost));
	wbTime_stop(Copy, "Copying output memory to the CPU");

	wbTime_start(GPU, "Freeing GPU Memory");
	cudaFree(deviceInput);
	cudaFree(deviceOutput);
	wbTime_stop(GPU, "Freeing GPU Memory");

	wbSolution(args, hostOutput, numElements);

	free(hostInput);
	cudaFreeHost(hostOutput);

#if LAB_DEBUG
	system("pause");
#endif

	return 0;
}