#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#define ITER 10000000

__global__ void setup_kernel(curandState *state){

  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  curand_init(1234, idx, 0, &state[idx]);
}

__global__ void generate_kernel(curandState *my_curandstate, const unsigned int n, double *result){

  int idx = threadIdx.x + blockDim.x*blockIdx.x;

  int count = 0;
  while (count < n){
    double myrandf = curand_uniform_double(my_curandstate+idx);
    if (myrandf < 0.000000001) {
        result[idx]++;
    }
    count++;}
}

int main(){

  curandState *d_state;
  cudaMalloc(&d_state, 32*sizeof(curandState));
  double *d_result, *h_result;
  cudaMalloc(&d_result, 32*sizeof(double));
  h_result = (double *)malloc(32*sizeof(double));
  cudaMemset(d_result, 0, 32*sizeof(double));
  setup_kernel<<<1,32>>>(d_state);

  generate_kernel<<<1,32>>>(d_state, ITER, d_result);
  cudaMemcpy(h_result, d_result, 32*sizeof(double), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 32; i++)
    printf("%d    %f\n", i, h_result[i]);

  return 0;
}
