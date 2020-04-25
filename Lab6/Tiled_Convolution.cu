#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
	    }                                                                     \
    } while (0)

#define Mask_width 5
#define Mask_radius Mask_width / 2
#define TILE_WIDTH 16
#define w (TILE_WIDTH + Mask_width - 1)
#define O_TILE_WIDTH 12
#define clamp(x) (min(max((x), 0.0), 1.0))

//TODO: INSERT CODE HERE

__global__ void convolutionTiled(float *I, const float *__restrict__ M,
	float *P, int channels, int width, int height) {
	

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int Col = blockIdx.x * Mask_width + threadIdx.x;
	int Row = blockIdx.y * Mask_width + threadIdx.y;

	//From Notes --> 8.3
	int col_o = blockIdx.x * O_TILE_WIDTH + threadIdx.x;
	int row_o = blockIdx.y * O_TILE_WIDTH + threadIdx.y;

	//int row_i = row_o - 2; -->Doesn't seet to work
	//int col_i = col_o - 2; -->Doesn't seem to work

	int radius = Mask_width / 2;



	__shared__ float shareMem[w][w];

	for (int x = 0; x < channels; x++){ //Apply mask and image channel for each pixel 

		//From Notes --> 8.3
		int xoffset = row_o - radius;
		int yoffset = col_o - radius;

		if ((xoffset >= 0) && (xoffset < height) && (yoffset >= 0) && (yoffset < width)){
			shareMem[ty][tx] = I[(xoffset * width + yoffset)*channels + x];  //Use width since patch is OK since pitch is set to width  
		}
		else{
			shareMem[ty][tx] = 0.0;
		}

		__syncthreads();


		//From Notes --> 8.3
		float acumVal = 0;
		int m, n;
		if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH){
			for (m  = 0; m < Mask_width; m++){
				for (int n = 0; n < Mask_width; n++){
					int rows = ty + m;
					int columns = tx + n;
					acumVal += shareMem[ty + m][tx + n] * M[m * Mask_width + n];
				}
			}
			
			if ((row_o < height) && (col_o < width)){
				P[(row_o * width + col_o)*channels + x] = clamp(acumVal); //Used from Convulation
			}

			__syncthreads();

		}		
		
	}
}


int main(int argc, char *argv[]) {
	wbArg_t arg;
	int maskRows;
	int maskColumns;
	int imageChannels;
	int imageWidth;
	int imageHeight;
	char *inputImageFile;
	char *inputMaskFile;
	wbImage_t inputImage;
	wbImage_t outputImage;
	float *hostInputImageData;
	float *hostOutputImageData;
	float *hostMaskData;
	float *deviceInputImageData;
	float *deviceOutputImageData;
	float *deviceMaskData;

	arg = wbArg_read(argc, argv); /* parse the input arguments */

	inputImageFile = wbArg_getInputFile(arg, 0);
	inputMaskFile = wbArg_getInputFile(arg, 1);

	inputImage = wbImport(inputImageFile);
	hostMaskData = (float *)wbImport(inputMaskFile, &maskRows, &maskColumns);

	assert(maskRows == 5);    /* mask height is fixed to 5 in this mp */
	assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

	imageWidth = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);
	imageChannels = wbImage_getChannels(inputImage);

	outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

	hostInputImageData = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);

	wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

	wbTime_start(GPU, "Doing GPU memory allocation");
	//TODO: INSERT CODE HERE
	cudaMalloc((void**)&deviceInputImageData, imageHeight * imageWidth * imageChannels * sizeof(float));
	cudaMalloc((void**)&deviceOutputImageData, imageHeight * imageWidth * imageChannels * sizeof(float));
	cudaMalloc((void**)&deviceMaskData, maskColumns * maskRows * sizeof(float));

	wbTime_stop(GPU, "Doing GPU memory allocation");

	wbTime_start(Copy, "Copying data to the GPU");
	//TODO: INSERT CODE HERE
	cudaMemcpy(deviceInputImageData, hostInputImageData, imageHeight * imageWidth * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceMaskData, hostMaskData, maskColumns * maskRows * sizeof(float), cudaMemcpyHostToDevice);

	wbTime_stop(Copy, "Copying data to the GPU");

	wbTime_start(Compute, "Doing the computation on the GPU");
	//TODO: INSERT CODE HERE
	dim3 gridSize(ceil((float)imageWidth / O_TILE_WIDTH), ceil((float)imageHeight / O_TILE_WIDTH));
	dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);
	convolutionTiled << < gridSize, blockSize >> > (deviceInputImageData, deviceMaskData, deviceOutputImageData, imageChannels, imageWidth, imageHeight);
	
	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Doing the computation on the GPU");

	wbTime_start(Copy, "Copying data from the GPU");
	//TODO: INSERT CODE HERE
	cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth*imageHeight*imageChannels *sizeof(float), cudaMemcpyDeviceToHost);

	wbTime_stop(Copy, "Copying data from the GPU");

	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

	wbSolution(arg, outputImage);

	//TODO: RELEASE CUDA MEMORY
	cudaFree(deviceMaskData);
	cudaFree(deviceOutputImageData);
	cudaFree(deviceInputImageData);

	free(hostMaskData);
	wbImage_delete(outputImage);
	wbImage_delete(inputImage);

#if LAB_DEBUG
	system("pause");
#endif

	return 0;
}
