#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

inline void gpu_cuda_check(cudaError_t result) {
    if(result!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
template<class T> class GpuTestBuffer {
    T* pointer_=nullptr;
    std::size_t count_=0;
public:
    explicit GpuTestBuffer(std::size_t count):count_(count) {
        if(count) gpu_cuda_check(cudaMalloc(reinterpret_cast<void**>(&pointer_),count*sizeof(T)));
    }
    explicit GpuTestBuffer(const std::vector<T>& values):GpuTestBuffer(values.size()) {
        if(count_) gpu_cuda_check(cudaMemcpy(pointer_,values.data(),count_*sizeof(T),cudaMemcpyHostToDevice));
    }
    ~GpuTestBuffer(){if(pointer_)cudaFree(pointer_);}
    GpuTestBuffer(const GpuTestBuffer&)=delete;
    GpuTestBuffer& operator=(const GpuTestBuffer&)=delete;
    T* data(){return pointer_;}
    const T* data()const{return pointer_;}
    std::vector<T> download()const {
        std::vector<T> out(count_);
        if(count_)gpu_cuda_check(cudaMemcpy(out.data(),pointer_,count_*sizeof(T),cudaMemcpyDeviceToHost));
        return out;
    }
};
