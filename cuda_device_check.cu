#include <cuda_runtime.h>

#include <cstdio>

__global__ void increment(int* value) { ++(*value); }

int main() {
  int device_count = 0;
  cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count < 1) {
    std::fprintf(stderr, "No usable CUDA device: %s\n", cudaGetErrorString(status));
    return 1;
  }

  int* device_value = nullptr;
  int host_value = 41;
  status = cudaMalloc(&device_value, sizeof(int));
  if (status != cudaSuccess) {
    std::fprintf(stderr, "cudaMalloc: %s\n", cudaGetErrorString(status));
    return 2;
  }
  status = cudaMemcpy(device_value, &host_value, sizeof(int), cudaMemcpyHostToDevice);
  if (status == cudaSuccess) {
    increment<<<1, 1>>>(device_value);
    status = cudaGetLastError();
  }
  if (status == cudaSuccess) status = cudaDeviceSynchronize();
  if (status == cudaSuccess) status = cudaMemcpy(&host_value, device_value, sizeof(int), cudaMemcpyDeviceToHost);
  cudaError_t free_status = cudaFree(device_value);
  if (status != cudaSuccess || free_status != cudaSuccess || host_value != 42) {
    std::fprintf(stderr, "CUDA check failed: %s, result=%d\n", cudaGetErrorString(status), host_value);
    return 3;
  }

  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, 0);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "cudaGetDeviceProperties: %s\n", cudaGetErrorString(status));
    return 4;
  }
  std::printf("CUDA device check passed: %s, result=%d\n", properties.name, host_value);
  return 0;
}
