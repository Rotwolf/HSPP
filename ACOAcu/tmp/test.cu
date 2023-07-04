#include <stdio.h>

int main(void) {
        cudaDeviceProp deviceProp;
        if (cudaSuccess != cudaGetDeviceProperties(&deviceProp, 0)) {
                printf("Get device properties failed.\n");
                return 1;
        } else {
                printf("The warp size is %d.\n", deviceProp.warpSize);
                return 0;
        }
}