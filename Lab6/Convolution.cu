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
#define clamp(x) (min(max((x), 0.0), 1.0))

__global__ void convolution(float *I, const float *M,
	float *P, int channels, int width, int height) {

	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int radius = Mask_width / 2;
	int maskWidth = 5;

	if (Col < width && Row < height){
		for (int x = 0; x < channels; x++){  //apply mask and image channel for each pixel 
		
			float pixval = 0;
			int xoffset = Row - radius;  //x position
			int yoffset = Col - radius;  //y position

			for (int z = 0; z < maskWidth; z++){   // These for loops will iterate through the mask

				for (int y = 0; y < maskWidth; y++){
					int pixRow = xoffset + z;
					int pixCol = yoffset + y;

					if (pixRow >= 0 && pixRow < height && pixCol >= 0 && pixCol < width){  //will go into if statement if the pixel for mask in inside the image boundary 

						float newVal = I[(pixRow*width + pixCol)*channels + x] * M[(z *radius + y)];
						pixval += newVal;

					}

				}
			}
			P[(Row*width + Col) * channels + x] = clamp(pixval);
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
	dim3 gridSize(ceil((float)imageWidth / 16), ceil((float)imageHeight / 16));
	dim3 blockSize(16,16);
	convolution<< < gridSize, blockSize >> > (deviceInputImageData, deviceMaskData, deviceOutputImageData, imageChannels, imageWidth, imageHeight);

	
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
