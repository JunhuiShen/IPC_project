#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

inline void resident_cuda_check(cudaError_t result) {
    if(result!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
template<class T> class ResidentTestBuffer {
    T* pointer_=nullptr;
    std::size_t count_=0;
public:
    explicit ResidentTestBuffer(std::size_t count):count_(count) {
        if(count) resident_cuda_check(cudaMalloc(reinterpret_cast<void**>(&pointer_),count*sizeof(T)));
    }
    explicit ResidentTestBuffer(const std::vector<T>& values):ResidentTestBuffer(values.size()) {
        if(count_) resident_cuda_check(cudaMemcpy(pointer_,values.data(),count_*sizeof(T),cudaMemcpyHostToDevice));
    }
    ~ResidentTestBuffer(){if(pointer_)cudaFree(pointer_);}
    ResidentTestBuffer(const ResidentTestBuffer&)=delete;
    ResidentTestBuffer& operator=(const ResidentTestBuffer&)=delete;
    T* data(){return pointer_;}
    const T* data()const{return pointer_;}
    std::vector<T> download()const {
        std::vector<T> out(count_);
        if(count_)resident_cuda_check(cudaMemcpy(out.data(),pointer_,count_*sizeof(T),cudaMemcpyDeviceToHost));
        return out;
    }
};
