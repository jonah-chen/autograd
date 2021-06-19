#include "diff.hpp"
#include "abbrev.hpp"

typedef diff<float> dfloat;

__global__ void dosmt(dfloat* in, dfloat* out, int sz, bool* positive)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < sz)
    {
        out[i] = SIN(in[i]);
        positive[i] = out[i] < in[i];
    }
}

__global__ void vec_exp(dfloat* vec, int sz)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < sz)
        vec[i] = EXP(vec[i]);
}

__global__ void 

int main()
{
    dfloat *h_a, *d_a, *h_s, *d_s;
    bool *h_b, *d_b;


    cudaMalloc((void**)&d_a, sizeof(dfloat) * 1024);
    cudaMalloc((void**)&d_s, sizeof(dfloat) * 1024);
    cudaMalloc((void**)&d_b, sizeof(bool) * 1024);

    h_a = new dfloat[1024];
    h_s = new dfloat[1024];
    h_b = new bool[1024];

    for (int i = 0; i < 1024; ++i)
    {
        h_a[i] = dfloat((float)i, 1.0f);
    }

    cudaMemcpy(d_a, h_a, sizeof(dfloat)*1024, cudaMemcpyHostToDevice);

    dosmt<<<1, 1024>>>(d_a, d_s, 1024, d_b);

    cudaMemcpy(h_s, d_s, sizeof(dfloat)*1024, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, sizeof(bool)*1024, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < 1024; ++i)
    {
        std::cout << h_b[i] << h_s[i] << std::endl;
    }
    cudaFree(d_a);
    cudaFree(d_s);
    cudaFree(d_b);
    delete[] h_a;
    delete[] h_s;
    delete[] h_b;
    
    return 0;
}
