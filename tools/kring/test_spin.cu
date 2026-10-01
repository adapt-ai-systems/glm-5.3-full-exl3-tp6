// kring self-test: a few normal kernels, then a spin kernel that waits for a host flag (the "hang"),
// then more kernels queued behind it on the same stream. While it spins, spark-kring must report
// PARKED/RUNNING = spin_until_flag and the queued kernels behind it.
//   nvcc -o test_spin test_spin.cu && CUDA_INJECTION64_PATH=./libkring.so KRING_DIR=/tmp/kring-test ./test_spin 5
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

__global__ void warmup_add(float *x, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] += 1.f; }
__global__ void spin_until_flag(volatile int *flag) { while (*flag == 0) { __nanosleep(1000); } }
__global__ void after_the_hang(float *x) { x[0] *= 2.f; }

int main(int argc, char **argv) {
  int secs = argc > 1 ? atoi(argv[1]) : 5;
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)
  float *x; CK(cudaMalloc(&x, 1 << 20));
  int *flag; CK(cudaMallocManaged(&flag, sizeof(int)));  // GB10: coherent unified memory
  *flag = 0; int *dflag = flag;
  cudaStream_t s; cudaStreamCreate(&s);
  for (int i = 0; i < 50; i++) warmup_add<<<256, 256, 0, s>>>(x, 1 << 18);
  CK(cudaStreamSynchronize(s));
  usleep(300000);  // let the 100 ms activity flush deliver the warmup completions
  spin_until_flag<<<1, 1, 0, s>>>(dflag);
  for (int i = 0; i < 3; i++) after_the_hang<<<1, 1, 0, s>>>(x);
  printf("spinning for %d s (pid %d)\n", secs, getpid()); fflush(stdout);
  sleep(secs);
  *(volatile int *)flag = 1;
  cudaStreamSynchronize(s);
  printf("released, rc=%s\n", cudaGetErrorString(cudaGetLastError()));
  return 0;
}
