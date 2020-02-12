#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include<chrono>
#include<iostream>

#define Row  1024
#define Col 1024


void matrix_mul_cpu(int *M, int *N, int *P, int width)
{
    for(int i=0;i<width;i++)
        for(int j=0;j<width;j++)
        {
            int sum = 0;
            for(int k=0;k<width;k++)
            {
                int a = M[i*width+k];
                int b = N[k*width+j];
                sum += a*b;
            }
            P[i*width+j] = sum;
        }
}
 
__global__ void matrix_mul_gpu(int *M, int* N, int* P, int width)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
                
    int sum = 0;
    for(int k=0;k<width;k++)
    {
        int a = M[j*width+k];
        int b = N[k*width+i];
        sum += a*b;
    }
    P[j*width+i] = sum;
}
 
int main()
{

    int *A = (int *)malloc(sizeof(int) * Row * Col);
    int *B = (int *)malloc(sizeof(int) * Row * Col);
    int *C = (int *)malloc(sizeof(int) * Row * Col);

    //set value
    for (int i = 0; i < Row*Col; i++) {
        A[i] = 90;
        B[i] = 10;
    }

    auto start = std::chrono::high_resolution_clock::now();

    matrix_mul_cpu(A, B, C, Col);

    auto span_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-start);

    std::cout<<"gemm by cpu cost:"<<span_cpu.count()<<" ms.\n";


    //malloc device memory
    int *d_dataA, *d_dataB, *d_dataC;
    cudaMalloc((void**)&d_dataA, sizeof(int) *Row*Col);
    cudaMalloc((void**)&d_dataB, sizeof(int) *Row*Col);
    cudaMalloc((void**)&d_dataC, sizeof(int) *Row*Col);

                                                                
    cudaMemcpy(d_dataA, A, sizeof(int) * Row * Col, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataB, B, sizeof(int) * Row * Col, cudaMemcpyHostToDevice);
    
    //线程矩阵
    dim3 threadPerBlock(16, 16);
    //线程块矩阵
    dim3 blockNumber((Col+threadPerBlock.x-1)/ threadPerBlock.x, (Row+threadPerBlock.y-1)/ threadPerBlock.y );//(64,64)

    printf("Block(%d,%d)   Grid(%d,%d).\n", threadPerBlock.x, threadPerBlock.y, blockNumber.x, blockNumber.y);
    
    start = std::chrono::high_resolution_clock::now();
    matrix_mul_gpu << <blockNumber, threadPerBlock >> > (d_dataA, d_dataB, d_dataC, Col); 
    cudaMemcpy(C, d_dataC, sizeof(int) * Row * Col, cudaMemcpyDeviceToHost);
    auto span_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-start);
    
    std::cout<<"gemm by cuda cost:"<<span_gpu.count()<<" ms.\n";

    printf("gpu %.2fx faster than cpu. \n",float(span_cpu.count())/span_gpu.count());
                                                                                              
    //释放内存
    free(A);
    free(B);
    free(C);
    cudaFree(d_dataA);
    cudaFree(d_dataB);
    cudaFree(d_dataC);


    return 0;
}