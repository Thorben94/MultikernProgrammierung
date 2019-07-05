#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <climits>
#include <algorithm>

void readField(std::vector<uint8_t> &field, int height) {
  std::string inputLine;
  for (int i = 0; i <= height; i++)
  {
    std::getline(std::cin, inputLine);
    std::vector<uint8_t> tempvec;
    std::istringstream input(inputLine);
    std::string temp;

    while (std::getline(input, temp, ' '))
    {
      if (temp == "X")
        tempvec.emplace_back(1);
      else if (temp == "O")
        tempvec.emplace_back(0);
      else
        std::cerr << "symbol mismatch\n";
    }
    for (uint8_t i : tempvec) {
      field.emplace_back(i);
    }
  }
}

void writeField(std::vector<uint8_t> &v, int width) {
  int count = 0;
  for (uint8_t i : v)
  {
    count++;
    // std::cout << static_cast<int>(i) << " ";
    std::cout << ((i == 0) ? 'O' : 'X') << " ";
    if (count % width == 0)
      std::cout << std::endl;
  }
}

cudaError_t lifeWithCuda(uint8_t *field, unsigned int width, unsigned int height, unsigned int generation);

__global__ void lifeKernelaggregator(uint8_t *field, uint8_t *tempfield, unsigned int width, unsigned int height)
{
  for (int b = blockIdx.x; b < height; b += gridDim.x) { // rows are always exclusive to blocks, they can only grow to height, thus never out of the size of the memory segment
    for (int t = threadIdx.x; t < width; t += blockDim.x) // threads not growing past the size of width ensures threads not accessing memory where they shouldn't
    {
      // establish identity of cell 
      int cellnr = b * width + t;
      // calculate 1d aggregated neigbourhood (top + mid + bot) and drop in temp
      if (b == 0)
        tempfield[cellnr] = field[(width * (height - 1)) + t];
      else
        tempfield[cellnr] = field[cellnr - width];  // row on top

      tempfield[cellnr] += field[cellnr];

      if (b == (height - 1))
        tempfield[cellnr] += field[(0 * width) + t];
      else
        tempfield[cellnr] += field[cellnr + width]; //row below
      __syncthreads(); // obsolete?
    }
  }
}

__global__ void lifeKernel(uint8_t *field, uint8_t *tempfield, unsigned int width, unsigned int height)
{
  for (int b = blockIdx.x; b < height; b += gridDim.x) { // rows are always exclusive to blocks, they can only grow to height, thus never out of the size of the memory segment
    for (int t = threadIdx.x; t < width; t += blockDim.x) // threads not growing past the size of width ensures threads not accessing memory where they shouldn't
    {
      int cellnr = b * width + t;
      // calculate cell value
      uint8_t left;
      if (t == 0)
        left = tempfield[cellnr + width - 1];
      else
        left = tempfield[cellnr - 1];
      uint8_t mid = tempfield[cellnr];
      uint8_t right;
      if (t == width - 1)
        right = tempfield[cellnr + 1 - width];
      else
        right = tempfield[cellnr + 1];

      if (field[cellnr] == 1) {
        if (!(3 <= left + mid + right && left + mid + right <= 4))
          field[cellnr] = 0;
      }
      else
      {
        if (left + mid + right == 3)
          field[cellnr] = 1;
      }
      __syncthreads();
    }
  }
}

int main()
{
  int generations;
  std::cin >> generations;
  int width;
  std::cin >> width;
  int height;
  std::cin >> height;
  std::vector<uint8_t> field; //( width, std::vector<float> ( height, 0 ) ) //could initialize

  readField(field, height);

  // generate generation X of game of life on field
  cudaError_t cudaStatus = lifeWithCuda(field.data(), width, height, generations);
  if (cudaStatus != cudaSuccess) {
    std::cerr << "lifeWithCuda failed!\n";
    return 1;
  }

  writeField(field, width);

  // cudaDeviceReset must be called before exiting in order for profiling and
  // tracing tools such as Nsight and Visual Profiler to show complete traces.
  cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
    std::cerr << "cudaDeviceReset failed!\n";
    return 1;
  }

  return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t lifeWithCuda(uint8_t *field, unsigned int width, unsigned int height, unsigned int generation)
{
  uint8_t *dev_field = 0;
  uint8_t *dev_field_temp = 0;
  cudaError_t cudaStatus;

  
  // for robustness against enormous entries
  unsigned int maxThreadsperBlock = 1024; // for my device maxThreadsperBlock is 1024
  unsigned int maxBlocksperGrid = 12288;   // max gridx is 2147483647, but shared memory per block is just 49152 Bytes, i assume 3 times my entry as memory usage, so i pick ~1/4th to be safe
  int threadnum = std::min(maxThreadsperBlock, width); 
  int blocknum = std::min(maxBlocksperGrid, height); 

  // Choose which GPU to run on, change this on a multi-GPU system.
  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    std::cerr << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n";
    goto Error;
  }

  // Allocate GPU buffers for two fields (one input, one output)    .
  cudaStatus = cudaMalloc((void**)&dev_field, height * (width * sizeof(uint8_t)));
  if (cudaStatus != cudaSuccess) {
    std::cerr << "cudaMalloc failed!\n";
    goto Error;
  }

  cudaStatus = cudaMalloc((void**)&dev_field_temp, height * (width * sizeof(uint8_t)));
  if (cudaStatus != cudaSuccess) {
    std::cerr << "cudaMalloc failed!\n";
    goto Error;
  }

  // Copy input field from host memory to GPU buffers.
  cudaStatus = cudaMemcpy(dev_field, field, height * (width * sizeof(uint8_t)), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    std::cerr << "cudaMemcpy failed!\n";
    goto Error;
  }

  // TODO: figure aout, what kind of thread distribution works best
  for (unsigned int g = 0; g < generation; g++)
  {
    lifeKernelaggregator <<<blocknum, threadnum >>> (dev_field, dev_field_temp, width, height);
    lifeKernel <<<blocknum, threadnum >>> (dev_field, dev_field_temp, width, height);
    std::cout << ".";
  }
  std::cout << std::endl;

  // Check for any errors launching the kernel
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    std::cerr << "addKernel launch failed: %s\n" << cudaGetErrorString(cudaStatus);
    goto Error;
  }

  // cudaDeviceSynchronize waits for the kernel to finish, and returns
  // any errors encountered during the launch.
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    std::cerr << "cudaDeviceSynchronize returned error code %d after launching addKernel!\n" << cudaStatus;
    goto Error;
  }

  // Copy output vector from GPU buffer to host memory.
  cudaStatus = cudaMemcpy(field, dev_field, height * (width * sizeof(uint8_t)), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    std::cerr << "cudaMemcpy failed!\n";
    goto Error;
  }

Error:
  cudaFree(dev_field);
  cudaFree(dev_field_temp);

  return cudaStatus;
}
