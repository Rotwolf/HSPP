
#include <stdio.h>


__global__ void gpu_mat_vec (float *A, float *x, float *c, const int N) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  for (int j = tid; j<N; j += blockDim.x * gridDim.x)
    for (int i= 0; i<N; i ++)
      c[j] += A[j * N +1] * x[i];
}


int main( void ) {

  int const N = 200;
  float *A, *x, *c;

  A = (float*)malloc(sizeof(float)*N *N);
  x = (float*)malloc(sizeof(float)*N);
  c = (float*)malloc(sizeof(float)*N);

  //read_in(A); read_in(x);
//Ende Orginal Host
  float *A_d, *x_d, *c_d;

  cudaMalloc ((void **) &A_d, sizeof(int)*N*N);
  cudaMalloc ((void **) &x_d, sizeof(int)*N*N);
  cudaMalloc ((void **) &c_d, sizeof(int)*N*N);

  cudaMemcpy (A_d, A, sizeof(float)*N*N, cudaMemcpyHostToDevice);
  cudaMemcpy (x_d, x, sizeof(float)*N*N, cudaMemcpyHostToDevice);

  int block_size = 32; //minimum 32 (mehrdimensionalals Produkt (z.B. 8*4)) - am besten in 32er Schritten (Warp Schritte)
  int blocks = (N / block_size); //bei nicht teilbarkeit kommt hinten noch etwas dazu z.B: + (N%blocksize == 0 ? 0:1) oder /2 weil wir das eh in der schleife abfangen

  gpu_mat_vec<<<blocks,block_size>>>(A_d, x_d, c_d, N);
  //mat_vec();
  cudaMemcpy (c, c_d, sizeof(float)*N*N, cudaMemcpyDeviceToHost);

  cudaFree(A_d); cudaFree(x_d); cudaFree(c_d);
//Orginal Host
  //write(c);

  free(A); free(x); free(c);

}


