#pragma once
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <limits>

namespace ipc_gpu::solver {
inline void cuda_checked(cudaError_t error) {
    if(error != cudaSuccess)
        throw std::runtime_error(std::string("GPU CUDA: ")+cudaGetErrorString(error));
}
template<class T> class DeviceArray {
    T* pointer_ = nullptr;
    std::size_t size_ = 0;
public:
    DeviceArray() = default;
    explicit DeviceArray(std::size_t n) { resize(n); }
    explicit DeviceArray(const std::vector<T>& values) { upload(values); }
    ~DeviceArray() { if(pointer_) cudaFree(pointer_); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    void resize(std::size_t n) {
        if(n==size_) return;
        if(n > std::numeric_limits<std::size_t>::max()/sizeof(T))
            throw std::overflow_error("GPU device allocation size");
        T* replacement=nullptr;
        if(n) cuda_checked(cudaMalloc(reinterpret_cast<void**>(&replacement),n*sizeof(T)));
        if(pointer_) cuda_checked(cudaFree(pointer_));
        pointer_=replacement;size_=n;
    }
    void upload(const std::vector<T>& values) {
        resize(values.size());
        if(size_) cuda_checked(cudaMemcpy(pointer_,values.data(),size_*sizeof(T),cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> values(size_);
        if(size_) cuda_checked(cudaMemcpy(values.data(),pointer_,size_*sizeof(T),cudaMemcpyDeviceToHost));
        return values;
    }
    void clear() { if(size_) cuda_checked(cudaMemset(pointer_,0,size_*sizeof(T))); }
    T* data() { return pointer_; }
    const T* data() const { return pointer_; }
    std::size_t size() const { return size_; }
};
} // namespace ipc_gpu::solver
