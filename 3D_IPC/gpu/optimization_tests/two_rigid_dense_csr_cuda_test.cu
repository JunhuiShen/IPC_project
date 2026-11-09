#define main dense_policy_host_main
#include "two_rigid_dense_csr_host_test.cpp"
#undef main
#include <cuda_runtime.h>
namespace {
void check(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
template<class T>struct Device {
    T* p=nullptr;std::size_t n;
    explicit Device(std::size_t count):n(count){if(n)check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));}
    explicit Device(const std::vector<T>& a):Device(a.size()){if(n)check(cudaMemcpy(p,a.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
    ~Device(){if(p)cudaFree(p);}
    Device(const Device&)=delete;Device& operator=(const Device&)=delete;
    std::vector<T> read()const{std::vector<T>a(n);if(n)check(cudaMemcpy(a.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return a;}
};
__global__ void inspect_contacts(const Contact* contacts,std::size_t count,
    const int* blocks,std::size_t nodes,unsigned* unsupported) {
    const auto pair=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(pair>=count)return;
    const int previous=pair?contacts[pair-1].kind:-1;
    const auto reason=dense::inspect(contacts[pair],previous,blocks,nodes);
    if(reason)atomicOr(unsupported,reason);
}
// Exact integer body of production emit_two_rigid_dense_rows; source proof
// binds this copy. Host generic_rows supplies an independent sorted oracle.
__global__ void dense_rows(std::size_t count,int* pairs,std::size_t* offsets) {
    const auto pair=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(pair==0){offsets[0]=0;offsets[1]=count;offsets[2]=2*count;}
    if(pair<count){pairs[pair]=static_cast<int>(pair);pairs[count+pair]=static_cast<int>(pair);}
}
}
int main() {
    try {
        require(dense_policy_host_main()==0,"host oracle failed");
        Device<unsigned> reason(1);
        for(const auto& c:dense_cases()) {
            Device<Contact> contacts(c.contacts);Device<int> blocks(c.blocks);
            check(cudaMemset(reason.p,0,sizeof(unsigned)));
            const unsigned grid=static_cast<unsigned>(std::max<std::size_t>(1,(c.contacts.size()+127)/128));
            inspect_contacts<<<grid,128>>>(contacts.p,c.contacts.size(),blocks.p,c.blocks.size(),reason.p);
            check(cudaGetLastError());const auto actual=reason.read();
            require(actual[0]==c.reason,c.name+" device unsupported reason differs");
            if(!actual[0]) {
                Device<int> pairs(2*c.contacts.size());Device<std::size_t> offsets(3);
                dense_rows<<<grid,128>>>(c.contacts.size(),pairs.p,offsets.p);check(cudaGetLastError());
                const auto columns=pairs.read();const auto off=offsets.read();
                require(off==std::vector<std::size_t>({0,c.contacts.size(),2*c.contacts.size()}),c.name+" dense offsets differ");
                const auto reference=generic_rows(c);
                for(unsigned block=0;block<2;++block)
                    require(std::vector<int>(columns.begin()+off[block],columns.begin()+off[block+1])==reference[block],c.name+" consumed GPU row differs");
            }
            const auto after=contacts.read();
            require(after.size()==c.contacts.size()&&(after.empty()||std::memcmp(after.data(),c.contacts.data(),after.size()*sizeof(Contact))==0),c.name+" GPU contacts mutated");
            require(blocks.read()==c.blocks,c.name+" GPU block map mutated");
        }
        check(cudaDeviceSynchronize());
        std::cout<<"Two-rigid dense CSR GPU integer preflight and consumed rows PASS: "<<dense_cases().size()<<" layouts\n";return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
