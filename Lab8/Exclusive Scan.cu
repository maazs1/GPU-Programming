#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <wb.h>

#define BLOCK_SIZE 512 //TODO: You can change this


#define TRANSPOSE_TILE_DIM 16
#define TRANSPOSE_BLOCK_ROWS 4

#define wbCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
// TODO: write kernel to uniform add each aux array value to corresponding block output
__global__ void uniformAdd(float* g_odata, float* g_aux, int n){
	int ID = blockIdx.x*blockDim.x + threadIdx.x;
	if (ID < n){
		g_odata[ID] += g_aux[blockIdx.x];
	}
}

// TODO: write a simple transpose kernel here
__global__ void transposeKernel(float *g_output, const float *g_input, int width, int height){
	__shared__ float sharedMem[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM];

	int x = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.y;
	int index_in = x + (y*height);
	x = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.x;
	y = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.y;
	int index_out = x + (y*height);

	for (int j = 0; j < TRANSPOSE_TILE_DIM; j += TRANSPOSE_BLOCK_ROWS){
		if (x < width){
			if ((y + j) < height){
				sharedMem[threadIdx.y + j][threadIdx.x] = g_input[index_in+j*width];
				__syncthreads();
			}
		}
	}
	//__syncthreads();

	//x = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.x;
	//y = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.y;

	for (int j = 0; j < TRANSPOSE_TILE_DIM; j += TRANSPOSE_BLOCK_ROWS)
		if (x < height){
			if ((y + j) < width){
				g_output[index_out+j*height] = sharedMem[threadIdx.x][threadIdx.y + j];
				__syncthreads();
			}
		}
}


// TODO: write 1D scan kernel here
__global__ void scan(float *g_odata, float *g_idata, float *g_aux, int n){
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

// TODO: write recursive scan wrapper on CPU here
void recursiveScan(float *g_odata, float *g_idata, int len){
	dim3 block(BLOCK_SIZE, 1);
	int numBlocks = (len / BLOCK_SIZE) + 1;
	dim3 grid(numBlocks, 1);
	float* g_aux;

	if (numBlocks == 1){
		g_aux = NULL;
		scan << <grid, block >> >(g_odata, g_idata, g_aux, len);
		cudaDeviceSynchronize();
	}
	else{
		cudaMalloc((void**)&g_aux, (numBlocks*sizeof(float)));
		float *secondValue;
		cudaMalloc((void**)&secondValue, (numBlocks*sizeof(float)));
		scan << <grid, block >> >(g_odata, g_idata, g_aux, len);
		cudaDeviceSynchronize();
		scan << <grid, block >> >(secondValue, g_aux, NULL, numBlocks);
		cudaDeviceSynchronize();
		recursiveScan(secondValue, g_aux, numBlocks);
		uniformAdd << <grid, block >> >(g_odata, secondValue, len);
		cudaDeviceSynchronize();
		cudaFree(g_aux);
		cudaFree(secondValue);
	}
}

int main(int argc, char **argv) {

	wbArg_t args;
	float *hostInput;  // The input 1D list
	float *hostOutput; // The output list
	float *deviceInput;  // device input
	float *deviceTmpOutput;  // temporary output
	float *deviceOutput;  // ouput 
	int numInputRows, numInputCols; // dimensions of the array

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numInputRows, &numInputCols);
	cudaHostAlloc(&hostOutput, numInputRows * numInputCols * sizeof(float),
		cudaHostAllocDefault);
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The dimensions of input are ",
		numInputRows, "x", numInputCols);

	wbTime_start(GPU, "Allocating GPU memory.");
	wbCheck(cudaMalloc((void **)&deviceInput, numInputRows * numInputCols * sizeof(float)));
	wbCheck(cudaMalloc((void **)&deviceOutput, numInputRows * numInputCols * sizeof(float)));
	wbCheck(cudaMalloc((void **)&deviceTmpOutput, numInputRows * numInputCols * sizeof(float)));
	wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(GPU, "Clearing output memory.");
	wbCheck(cudaMemset(deviceOutput, 0, numInputRows * numInputCols * sizeof(float)));
	wbTime_stop(GPU, "Clearing output memory.");

	wbTime_start(GPU, "Copying input memory to the GPU.");
	wbCheck(cudaMemcpy(deviceInput, hostInput, numInputRows * numInputCols * sizeof(float),
		cudaMemcpyHostToDevice));
	wbTime_stop(GPU, "Copying input memory to the GPU.");

	wbTime_start(Compute, "Performing CUDA computation");
	//TODO: Modify this to complete the functionality of the scan on the deivce
	for (int i = 0; i < numInputRows; ++i) {
		// TODO: call your 1d scan kernel for each row here
		int row = i*numInputCols;
		recursiveScan(deviceTmpOutput + row, deviceInput + row, numInputCols);
		cudaDeviceSynchronize();
	}

	// You can change TranposeBlockDim and TranposeGridDim, but if you use kernel suggested in the manual file, these should be the correct ones
	dim3 transposeBlockDim(TRANSPOSE_TILE_DIM, TRANSPOSE_BLOCK_ROWS);
	dim3 transposeGridDim(ceil(numInputCols / (float)TRANSPOSE_TILE_DIM), ceil(numInputRows / (float)TRANSPOSE_TILE_DIM));
	// TODO: call your transpose kernel here
	transposeKernel << <transposeGridDim, transposeBlockDim >> >(deviceOutput, deviceTmpOutput, numInputRows, numInputCols);
	cudaDeviceSynchronize();

	for (int i = 0; i < numInputCols; ++i) {
		// TODO: call your 1d scan kernel for each row of the tranposed matrix here
		int col = i*numInputRows;
		recursiveScan(deviceTmpOutput + col, deviceOutput + col, numInputCols);
		cudaDeviceSynchronize();
	}

	// You can change TranposeBlockDim and TranposeGridDim, but if you use kernel suggested in the manual file, these should be the correct ones
	transposeBlockDim = dim3(TRANSPOSE_TILE_DIM, TRANSPOSE_BLOCK_ROWS);
	transposeGridDim = dim3(ceil(numInputRows / (float)TRANSPOSE_TILE_DIM), ceil(numInputCols / (float)TRANSPOSE_TILE_DIM));
	// TODO: call your transpose kernel to get the final result here
	transposeKernel << <transposeGridDim, transposeBlockDim >> >(deviceOutput, deviceTmpOutput, numInputRows, numInputCols);
	cudaDeviceSynchronize();

	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output memory to the CPU");
	wbCheck(cudaMemcpy(hostOutput, deviceOutput, numInputRows * numInputCols * sizeof(float),
		cudaMemcpyDeviceToHost));
	wbTime_stop(Copy, "Copying output memory to the CPU");

	wbTime_start(GPU, "Freeing GPU Memory");
	cudaFree(deviceInput);
	cudaFree(deviceTmpOutput);
	cudaFree(deviceOutput);
	wbTime_stop(GPU, "Freeing GPU Memory");

	wbSolution(args, hostOutput, numInputRows, numInputCols);

	free(hostInput);
	cudaFreeHost(hostOutput);

	wbCheck(cudaDeviceSynchronize());

#if LAB_DEBUG
	system("pause");
#endif

	return 0;
}