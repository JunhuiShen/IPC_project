// nvcc -std=c++17 -O2 --fmad=false -I gpu \
//   gpu/contact_cooperative_test.cu gpu/contact_cooperative.cu \
//   -o /tmp/contact_cooperative_cuda_test
#define IPC_CONTACT_COOPERATIVE_CUDA_TEST
#include "contact_cooperative_test.cpp"
#include "contact_cooperative.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <new>
#include <numeric>

namespace {
namespace exact = ipc_gpu::exact;
namespace coop = ipc_gpu::solver::contact_cooperative_detail;
namespace ced = ipc_gpu::solver::contact_exact_detail;
namespace cd = ipc_gpu::solver::contact_detail;
using cooperative_test::require;
void cuda_check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
template<class T> struct Device {
    T* data = nullptr; std::size_t size;
    explicit Device(std::size_t n) : size(n) { if (n) cuda_check(cudaMalloc(&data, n*sizeof(T))); }
    explicit Device(const std::vector<T>& values) : Device(values.size()) { upload(values); }
    ~Device() { cudaFree(data); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    void upload(const std::vector<T>& values) {
        require(values.size() == size, "device upload size");
        if (size) cuda_check(cudaMemcpy(data, values.data(), size*sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> values(size);
        if (size) cuda_check(cudaMemcpy(values.data(), data, size*sizeof(T), cudaMemcpyDeviceToHost));
        return values;
    }
};

__global__ void reference_kernel(const ContactInput* input, ContactOutput* small,
    ContactOutput* complete, unsigned count, std::uint32_t* words, int* slots, int* failure) {
    exact::DevicePool pool;
    exact::initialize_pool(pool, words, slots, 1600, 512);
    for (unsigned i = 0; i < count; ++i) {
        small[i] = evaluate_contact_exact<256>(input[i]);
        complete[i] = small[i].status == ContactStatus::ExactCapacityExceeded
            ? evaluate_contact_exact<1600>(input[i]) : small[i];
        if (pool.failed || pool.available != 512) *failure = 1;
    }
    exact::release_pool();
}

__global__ void retry_kernel(const ContactInput* input, ContactOutput* output,
    const int* retries, unsigned count, std::uint32_t* words, int* slots, int* failure) {
    exact::DevicePool pool;
    exact::initialize_pool(pool, words, slots, 1600, 512);
    for (unsigned i = 0; i < count; ++i) {
        output[retries[i]] = evaluate_contact_exact<1600>(input[retries[i]]);
        if (pool.failed || pool.available != 512) *failure = 1;
    }
    exact::release_pool();
}

__device__ std::uint64_t geometry_hash(const coop::SharedGeometry& g) {
    const coop::IntegerView views[] = {g.residual[0],g.residual[1],g.residual[2],
        g.squared_residual,g.normal_den,g.curvature_den,g.z,g.ww,
        g.gram[0][0],g.gram[0][1],g.gram[1][0],g.gram[1][1],
        g.derivatives[0][0],g.derivatives[0][1],g.derivatives[1][0],g.derivatives[1][1],
        g.derivatives[2][0],g.derivatives[2][1]};
    std::uint64_t hash = 1469598103934665603ULL;
    for (const auto& v : views) {
        hash = (hash ^ std::uint64_t(v.used))*1099511628211ULL;
        hash = (hash ^ std::uint64_t(v.sign))*1099511628211ULL;
        for (int i = 0; i < v.used; ++i) hash = (hash ^ v.limb[i])*1099511628211ULL;
    }
    return hash;
}

__device__ std::uint64_t conversion_hash(const coop::SharedConversions& g) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (int component = 0; component < 8; ++component) {
        const auto& v = component < 7 ? g.numerator[component] : g.denominator;
        hash = (hash ^ std::uint64_t(v.used))*1099511628211ULL;
        hash = (hash ^ std::uint64_t(v.sign))*1099511628211ULL;
        hash = (hash ^ std::uint64_t(v.valid))*1099511628211ULL;
        for (int i = 0; i < v.used; ++i) hash = (hash ^ v.limb[i])*1099511628211ULL;
    }
    return hash;
}

// Deliberately tiny helper pools prove that adapters neither modify owner
// limbs nor return the owner's slots. Failed helpers leave no live Integer;
// the untouched original evaluator remains usable after owner destruction.
__global__ void ownership_kernel(const ContactInput* input, std::uint32_t* words,
    int* slots, unsigned helper_limit, int* diagnostics, bool conversions) {
    const unsigned lane = threadIdx.x;
    __shared__ coop::SharedGeometry shared;
    __shared__ coop::SharedConversions shared_conversions;
    __shared__ std::uint64_t original_hash;
    __shared__ unsigned prepared;
    exact::DevicePool pool;
    const unsigned offset = lane == 0 ? 0 : coop::owner_slots+(lane-1)*coop::helper_slots;
    const int limit = lane == 0 ? coop::owner_slots : int(helper_limit);
    if (lane < 9)
        exact::initialize_pool(pool, words+std::size_t(offset)*coop::capacity,
            slots+offset, coop::capacity, limit);
    using Owner = coop::GeometryOwner<256>;
    alignas(Owner) unsigned char storage[sizeof(Owner)];
    Owner* owner = nullptr;
    ContactOutput output;
    if (lane == 0) {
        owner = new (storage) Owner{};
        if (conversions) {
            prepared = coop::prepare_feature(*input, *owner, output);
            if (prepared) coop::publish_conversions(*owner, shared_conversions);
        } else {
            prepared = coop::prepare_geometry(*input, coop::self_block(input->hessian_blocks), *owner, output, shared);
        }
        if (!prepared || pool.failed) atomicOr(diagnostics, 1);
        original_hash = !prepared ? 0 : conversions ? conversion_hash(shared_conversions) : geometry_hash(shared);
    }
    __syncthreads();
    if (prepared && lane > 0 && lane < (conversions ? 8u : 9u)) {
        const auto result = conversions
            ? coop::convert_geometry_component<256>(shared_conversions, int(lane)-1)
            : coop::compute_entry<256>(shared, lane);
        if (!result.ok || pool.failed) atomicAdd(diagnostics+1, 1);
        if (pool.available != limit) atomicOr(diagnostics, 2);
    }
    __syncthreads();
    if (lane == 0) {
        if (prepared && (conversions ? conversion_hash(shared_conversions) : geometry_hash(shared)) != original_hash)
            atomicOr(diagnostics, 4);
        owner->~Owner();
        if (pool.failed || pool.available != coop::owner_slots) atomicOr(diagnostics, 8);
        const auto fallback = evaluate_contact_exact<256>(*input);
        if (fallback.status != ContactStatus::Ok || !fallback.active || pool.failed
            || pool.available != coop::owner_slots) atomicOr(diagnostics, 16);
    }
    __syncthreads();
    if (lane < 9) exact::release_pool();
}

int expected_error(ContactStatus status) {
    if (status == ContactStatus::Ok || status == ContactStatus::ExactCapacityExceeded) return 0;
    if (status == ContactStatus::InvalidInput) return 20;
    if (status == ContactStatus::ZeroDistance) return 21;
    return 22;
}

bool ee_policy_same_double_bits(double lhs,double rhs) {
    return std::memcmp(&lhs,&rhs,sizeof(double)) == 0;
}

// Full payload comparison, deliberately bitwise: a policy mismatch in one
// Gram/determinant path must not be hidden by a tolerance.
bool ee_policy_same_output_bits(const ContactOutput& lhs,const ContactOutput& rhs) {
    if (!ee_policy_same_double_bits(lhs.distance,rhs.distance)) return false;
    for (int i=0;i<4;++i) if (!ee_policy_same_double_bits(lhs.weights[i],rhs.weights[i])) return false;
    for (int k=0;k<3;++k) {
        if (!ee_policy_same_double_bits(lhs.normal[k],rhs.normal[k])) return false;
        if (!ee_policy_same_double_bits(lhs.friction_relative_gradient[k],rhs.friction_relative_gradient[k])) return false;
    }
    for (int i=0;i<4;++i) for (int k=0;k<3;++k)
        if (!ee_policy_same_double_bits(lhs.barrier_gradient[i][k],rhs.barrier_gradient[i][k])) return false;
    for (int i=0;i<4;++i) for (int j=0;j<4;++j) for (int e=0;e<9;++e)
        if (!ee_policy_same_double_bits(lhs.barrier_hessian[i][j][e],rhs.barrier_hessian[i][j][e])) return false;
    for (int e=0;e<9;++e)
        if (!ee_policy_same_double_bits(lhs.friction_relative_hessian[e],rhs.friction_relative_hessian[e])) return false;
    return lhs.status == rhs.status && lhs.active == rhs.active;
}

struct EePolicyFixture {
    const char* name;
    double positions[4][3];
    double d_hat;
    double eps;
};

EePolicyFixture make_ee_fixture(const char* name,const double (&positions)[4][3],
    double d_hat=0.25,double eps=1e-12) {
    EePolicyFixture fixture;
    fixture.name=name;
    std::memcpy(fixture.positions,positions,sizeof(fixture.positions));
    fixture.d_hat=d_hat;
    fixture.eps=eps;
    return fixture;
}

std::vector<EePolicyFixture> ee_policy_fixtures() {
    std::vector<EePolicyFixture> fixtures;
    const double base[4][3]={{0,0,0},{0,1,0},{0,0.0625,0.125},{1,0.0625,0.125}};
    fixtures.push_back(make_ee_fixture("base_gap125",base));
    for (double gap : {0x1p-30,1e-12}) {
        double positions[4][3];
        std::memcpy(positions,base,sizeof(positions));
        positions[2][2]=positions[3][2]=gap;
        fixtures.push_back(make_ee_fixture(gap==0x1p-30?"base_gap2^-30":"base_gap1e-12",positions));
    }
    double reverse_a[4][3];
    std::memcpy(reverse_a,base,sizeof(reverse_a));
    for (int k=0;k<3;++k) { reverse_a[0][k]=base[1][k]; reverse_a[1][k]=base[0][k]; }
    fixtures.push_back(make_ee_fixture("reverse_a",reverse_a));
    double reverse_b[4][3];
    std::memcpy(reverse_b,base,sizeof(reverse_b));
    for (int k=0;k<3;++k) { reverse_b[2][k]=base[3][k]; reverse_b[3][k]=base[2][k]; }
    fixtures.push_back(make_ee_fixture("reverse_b",reverse_b));
    const double swapped[4][3]={{0,0.0625,0.125},{1,0.0625,0.125},{0,0,0},{0,1,0}};
    fixtures.push_back(make_ee_fixture("swap_edges",swapped));
    const double u=0x1p-52;
    const double rounded[4][3]={{0,0,0},{1+u,1,0},{-0.5-u/2,0.5+u,-0.01},{0.5-u/2,-0.5,-0.01}};
    fixtures.push_back(make_ee_fixture("rounded",rounded));
    const double outside[4][3]={{0,0,0},{1,0,0},{-0.1,-5e-15,-0.01},{-0.1,1-5e-15,-0.01}};
    fixtures.push_back(make_ee_fixture("outside",outside));
    const double robust_world[4][3]={{0x1p40,0,0},{0x1p40,1,0},
        {0x1p40,0.0625,0.001},{0x1p40+1,0.0625,0.001}};
    fixtures.push_back(make_ee_fixture("robust_world",robust_world));
    const double robust_gram[4][3]={{0,0,0},{0,1,0},
        {0,0.0625,0.125},{0x1p-25,1.0625,0.125}};
    fixtures.push_back(make_ee_fixture("robust_gram",robust_gram));
    return fixtures;
}

ContactInput ee_policy_input(const EePolicyFixture& fixture,unsigned blocks,double friction) {
    ContactInput in;
    in.kind=ContactKind::EdgeEdge;
    for (int i=0;i<4;++i) for (int k=0;k<3;++k) {
        in.positions[i][k]=fixture.positions[i][k];
        in.previous[i][k]=fixture.positions[i][k];
    }
    // Deterministic slip for nonzero friction; friction zero still exercises
    // the full zero-friction/status contract for the identical geometry.
    in.previous[1][0]+=3.0e-4;
    in.previous[1][1]+=7.0e-4;
    in.d_hat=fixture.d_hat;
    in.k_barrier=100.0;
    in.dt=1.0/300.0;
    in.friction=friction;
    in.eps_v=0.001;
    in.eps=fixture.eps;
    in.hessian_blocks=blocks;
    return in;
}

struct GenericEeBasis {
    bool valid=false;
    bool free_s=false;
    bool free_t=false;
};

// Generic exact edge/edge basis, before any CPU contact policy override.
template<int C> GenericEeBasis generic_ee_basis(const ContactInput& in) {
    exact::Arithmetic<C> a;
    int unit=0; bool have_unit=false;
    for (int i=0;i<4;++i) for (int k=0;k<3;++k) {
        const auto value=exact::decompose(in.positions[i][k]);
        if (!value.valid) return {};
        if (value.sign&&(!have_unit||value.exponent<unit)) { unit=value.exponent; have_unit=true; }
    }
    exact::Points<C> points;
    for (int i=0;i<4;++i) for (int k=0;k<3;++k) {
        points.p[i].v[k]=exact::from_double<C>(in.positions[i][k],unit);
        a.ok=a.ok&&points.p[i].v[k].valid;
    }
    if (!a.ok) return {};
    const auto feature=ced::edge_edge(a,points);
    if (!a.ok||!feature.have) return {};
    GenericEeBasis basis; basis.valid=true;
    for (int j=0;j<feature.free_parameters;++j) {
        if (feature.sigma[j][0]==-1&&feature.sigma[j][1]==1) basis.free_s=true;
        if (feature.sigma[j][2]==1&&feature.sigma[j][3]==-1) basis.free_t=true;
    }
    return basis;
}

// Host replay of the cooperative pipeline, including the kernel's scalar
// fallback decision. The scalar oracle below is the real preview evaluator.
template<int C> ContactOutput ee_policy_host_cooperative_one(const ContactInput& in,bool parallel) {
    const int role=coop::self_block(in.hessian_blocks);
    if (role<0) return evaluate_contact_exact<C>(in);
    coop::GeometryOwner<C> owner;
    ContactOutput out;
    coop::SharedGeometry geometry{};
    bool prepared;
    if (parallel) {
        prepared=coop::prepare_feature(in,owner,out);
        if (!prepared) return out.status==ContactStatus::Ok?out:evaluate_contact_exact<C>(in);
        coop::SharedConversions conversions{};
        coop::publish_conversions(owner,conversions);
        double values[7]{};
        bool okay=true;
        for (int component=0;component<7;++component) {
            const auto result=coop::convert_geometry_component<C>(conversions,component);
            values[component]=result.value;
            okay=okay&&result.ok;
        }
        if (!okay) return evaluate_contact_exact<C>(in);
        for (int k=0;k<3;++k) owner.residual[k]=values[k];
        for (int i=0;i<4;++i) out.weights[i]=values[3+i];
        prepared=coop::prepare_derivatives(in,role,owner,out,geometry);
        if (!prepared) return out.status==ContactStatus::Ok?out:evaluate_contact_exact<C>(in);
    } else {
        prepared=coop::prepare_geometry<C>(in,role,owner,out,geometry);
        if (!prepared) return out.status==ContactStatus::Ok?out:evaluate_contact_exact<C>(in);
    }
    for (int lane=0;lane<9;++lane) {
        const auto result=coop::compute_entry<C>(geometry,lane);
        out.barrier_hessian[role][role][lane]=result.value;
        if (!result.ok) return evaluate_contact_exact<C>(in);
    }
    coop::finish_output(in,owner,out);
    if (out.status!=ContactStatus::Ok) return evaluate_contact_exact<C>(in);
    return out;
}

std::size_t ee_policy_probe(const std::vector<EePolicyFixture>& fixtures) {
    std::size_t basis_changes=0, robust_controls=0;
    for (const auto& fixture : fixtures) {
        const auto in=ee_policy_input(fixture,1u,0.0);
        const auto cpu=cd::segment_contact_cpu_ordinary_feature(in.positions,in.eps);
        const auto generic=generic_ee_basis<256>(in);
        require(generic.valid,"generic exact EE feature must exist for policy fixture");
        if (cpu.regular) {
            if (cpu.free_s!=generic.free_s||cpu.free_t!=generic.free_t) ++basis_changes;
        } else ++robust_controls;
        if (!std::strcmp(fixture.name,"rounded"))
            require(cpu.regular&&cpu.free_s&&cpu.free_t,
                "rounded CPU ordinary fixture must stay inclusive Interior");
        if (!std::strcmp(fixture.name,"outside"))
            require(cpu.regular&&!cpu.free_s&&!cpu.free_t,
                "1e-14 outside fixture must use CPU corner policy");
        if (!std::strcmp(fixture.name,"robust_world")||!std::strcmp(fixture.name,"robust_gram"))
            require(!cpu.regular,"robust controls must keep the strict exact feature");
        if (!std::strcmp(fixture.name,"base_gap125")||!std::strcmp(fixture.name,"base_gap2^-30")
            ||!std::strcmp(fixture.name,"base_gap1e-12")||!std::strcmp(fixture.name,"reverse_a")
            ||!std::strcmp(fixture.name,"reverse_b")||!std::strcmp(fixture.name,"swap_edges"))
            require(cpu.regular&&cpu.free_s&&cpu.free_t
                &&(generic.free_s!=cpu.free_s||generic.free_t!=cpu.free_t),
                "ordinary EE fixture must change the derivative basis");
    }
    require(basis_changes>=8,"EE policy suite must perturb the generic basis");
    require(robust_controls==2,"EE policy suite must include both robust controls");
    return basis_changes;
}

std::vector<ContactInput> ee_policy_records(const std::vector<EePolicyFixture>& fixtures) {
    const double friction_values[2]={0.0,1.3};
    const unsigned masks[6]={1u<<0,1u<<5,1u<<10,1u<<15,0xffffu,0x21u};
    std::vector<ContactInput> records;
    records.reserve(fixtures.size()*2*6);
    for (const auto& fixture : fixtures) for (double friction : friction_values) for (unsigned mask : masks)
        records.push_back(ee_policy_input(fixture,mask,friction));
    return records;
}

std::size_t ee_policy_host_checks(const std::vector<ContactInput>& records) {
    std::size_t checks=0;
    for (const auto& in : records) {
        const auto scalar=evaluate_contact_exact<256>(in);
        require(scalar.status==ContactStatus::Ok,"EE policy host scalar status");
        require(scalar.active!=0,"EE policy host scalar must be active");
        const auto serial=ee_policy_host_cooperative_one<256>(in,false);
        const auto parallel=ee_policy_host_cooperative_one<256>(in,true);
        require(ee_policy_same_output_bits(serial,scalar),"host cooperative serial != scalar");
        require(ee_policy_same_output_bits(parallel,scalar),"host cooperative parallel != scalar");
        require((serial.status==ContactStatus::ExactCapacityExceeded)
            ==(scalar.status==ContactStatus::ExactCapacityExceeded),"host serial queue capacity");
        require((parallel.status==ContactStatus::ExactCapacityExceeded)
            ==(scalar.status==ContactStatus::ExactCapacityExceeded),"host parallel queue capacity");
        checks+=2;
    }
    return checks;
}

std::size_t ee_policy_cuda_checks(const std::vector<ContactInput>& records) {
    Device<ContactInput> inputs(records);
    Device<ContactOutput> small(records.size()),complete(records.size());
    Device<std::uint32_t> reference_words(1600*512);
    Device<int> reference_slots(512),diagnostics(std::vector<int>{0});
    reference_kernel<<<1,1>>>(inputs.data,small.data,complete.data,unsigned(records.size()),
        reference_words.data,reference_slots.data,diagnostics.data);
    cuda_check(cudaGetLastError());
    require(diagnostics.download()[0]==0,"EE policy scalar reference arena");
    const auto small_reference=small.download(),reference=complete.download();
    for (std::size_t i=0;i<records.size();++i)
        require(small_reference[i].status==ContactStatus::Ok&&small_reference[i].active!=0,
            "EE policy scalar device reference must use one capacity");
    std::vector<int> ids(records.size());
    std::iota(ids.begin(),ids.end(),0);
    ContactOutput sentinel; sentinel.distance=-11; sentinel.active=99;
    const std::vector<ContactOutput> sentinels(records.size(),sentinel);
    std::size_t checks=0;
    for (bool parallel : {false,true}) for (unsigned workers : {1u,3u,8u}) {
        const auto arena=cooperative_exact_arena_sizes(workers);
        Device<std::uint32_t> words(arena.words);
        Device<int> slots(arena.free_slots),queue(ids),retries(ids.size());
        Device<unsigned> count(std::vector<unsigned>{unsigned(records.size())}),
            retry_count(std::vector<unsigned>{0}),next(std::vector<unsigned>{0});
        Device<int> failed(std::vector<int>{0});
        Device<ContactOutput> outputs(sentinels);
        for (int repetition=0;repetition<2;++repetition) {
            auto order=ids;
            if (repetition) std::reverse(order.begin(),order.end());
            queue.upload(order); retry_count.upload({0}); next.upload({0});
            failed.upload({0}); outputs.upload(sentinels);
            launch_cooperative_exact_contacts(inputs.data,outputs.data,queue.data,count.data,
                retries.data,retry_count.data,next.data,words.data,slots.data,workers,failed.data,
                nullptr,nullptr,parallel);
            cuda_check(cudaGetLastError());
            require(failed.download()[0]==0,"EE policy cooperative launch failed");
            require(retry_count.download()[0]==0,"EE policy cooperative unexpected capacity retry");
            const auto actual=outputs.download();
            for (std::size_t i=0;i<records.size();++i) {
                require(ee_policy_same_output_bits(actual[i],reference[i]),"device cooperative EE policy != scalar");
                ++checks;
            }
        }
    }
    return checks;
}

void ee_policy_cases() {
    const auto fixtures=ee_policy_fixtures();
    require(fixtures.size()==10,"EE policy fixture count");
    const auto records=ee_policy_records(fixtures);
    const auto basis_changes=ee_policy_probe(fixtures);
    const auto host_checks=ee_policy_host_checks(records);
    const auto cuda_checks=ee_policy_cuda_checks(records);
    std::cout << "COOPERATIVE_EE_POLICY_COMPLETE fixtures=" << fixtures.size()
        << " records=" << records.size() << " host_checks=" << host_checks
        << " cuda_checks=" << cuda_checks << " basis_changes=" << basis_changes
        << " robust_controls=2 PASS\n";
}
} // namespace

void cooperative_cuda_cases(const std::vector<ContactInput>& records) {
    Device<ContactInput> inputs(records);
    Device<ContactOutput> small(records.size()), complete(records.size());
    Device<std::uint32_t> reference_words(1600*512);
    Device<int> reference_slots(512), diagnostics(std::vector<int>{0});
    reference_kernel<<<1,1>>>(inputs.data, small.data, complete.data, unsigned(records.size()),
        reference_words.data, reference_slots.data, diagnostics.data);
    cuda_check(cudaGetLastError());
    require(diagnostics.download()[0] == 0, "scalar device reference arena");
    const auto small_reference = small.download(), reference = complete.download();
    std::vector<int> okay;
    unsigned expected_retries = 0;
    for (std::size_t i = 0; i < records.size(); ++i) {
        if (reference[i].status == ContactStatus::Ok) {
            okay.push_back(int(i));
            expected_retries += small_reference[i].status == ContactStatus::ExactCapacityExceeded;
        }
    }
    require(!okay.empty() && expected_retries > 0, "fixtures must exercise success and large retries");
    ContactOutput sentinel; sentinel.distance = -7; sentinel.active = 77;
    const std::vector<ContactOutput> sentinels(records.size(), sentinel);
    for (bool parallel : {false,true}) for (unsigned workers : {1u,3u,17u,64u}) {
        const auto arena = cooperative_exact_arena_sizes(workers);
        Device<std::uint32_t> words(arena.words);
        Device<int> slots(arena.free_slots), queue(okay), retries(okay.size());
        Device<unsigned> count(std::vector<unsigned>{unsigned(okay.size())}), retry_count(std::vector<unsigned>{0}), next(std::vector<unsigned>{0});
        Device<int> failed(std::vector<int>{0});
        Device<ContactOutput> outputs(sentinels);
        for (int repetition = 0; repetition < 2; ++repetition) {
            auto order = okay;
            if (repetition) std::reverse(order.begin(), order.end());
            queue.upload(order); retry_count.upload({0}); next.upload({0}); failed.upload({0}); outputs.upload(sentinels);
            launch_cooperative_exact_contacts(inputs.data, outputs.data, queue.data, count.data,
                retries.data, retry_count.data, next.data, words.data, slots.data, workers, failed.data,
                nullptr, nullptr, parallel);
            cuda_check(cudaGetLastError());
            require(failed.download()[0] == 0, "cooperative valid queue failed");
            const auto retry_size = retry_count.download()[0];
            require(retry_size == expected_retries, "cooperative changed retry count");
            auto retry_ids = retries.download(); retry_ids.resize(retry_size);
            std::sort(retry_ids.begin(), retry_ids.end());
            std::vector<int> expected_ids;
            for (int id : okay)
                if (small_reference[id].status == ContactStatus::ExactCapacityExceeded) expected_ids.push_back(id);
            std::sort(expected_ids.begin(), expected_ids.end());
            require(retry_ids == expected_ids, "cooperative changed retry membership");
            const auto before_retry = outputs.download();
            for (int id : okay) {
                if (small_reference[id].status == ContactStatus::ExactCapacityExceeded)
                    cooperative_test::compare(before_retry[id], sentinel, "retry output must remain untouched");
                else cooperative_test::compare(before_retry[id], small_reference[id], "device cooperative small");
            }
            retry_kernel<<<1,1>>>(inputs.data, outputs.data, retries.data, retry_size,
                reference_words.data, reference_slots.data, diagnostics.data);
            cuda_check(cudaGetLastError());
            require(diagnostics.download()[0] == 0, "large retry arena");
            const auto actual = outputs.download();
            for (int id : okay) cooperative_test::compare(actual[id], reference[id], "device complete");
        }
        // Empty queue and pre-existing error must not touch output records.
        outputs.upload(sentinels); count.upload({0}); next.upload({0}); retry_count.upload({0});
        launch_cooperative_exact_contacts(inputs.data, outputs.data, queue.data, count.data,
            retries.data, retry_count.data, next.data, words.data, slots.data, workers, failed.data,
            nullptr, nullptr, parallel);
        cuda_check(cudaGetLastError());
        require(next.download()[0] == 0 && retry_count.download()[0] == 0, "empty queue changed counters");
        count.upload({unsigned(okay.size())}); failed.upload({123});
        launch_cooperative_exact_contacts(inputs.data, outputs.data, queue.data, count.data,
            retries.data, retry_count.data, next.data, words.data, slots.data, workers, failed.data,
            nullptr, nullptr, parallel);
        cuda_check(cudaGetLastError());
        require(failed.download()[0] == 123 && next.download()[0] == 0, "pre-existing error changed");
        const auto untouched = outputs.download();
        for (const auto& out : untouched) cooperative_test::compare(out, sentinel, "empty/pre-failed output");
        std::cout << "CUDA parallel=" << parallel << " workers=" << workers << " successful queue=" << okay.size()
            << " retries=" << expected_retries << " reused arenas PASS\n";
    }
    // One bad input per launch makes the original status/error mapping
    // deterministic, independent of which worker wins the dynamic queue.
    const auto arena = cooperative_exact_arena_sizes(1);
    Device<std::uint32_t> words(arena.words); Device<int> slots(arena.free_slots);
    Device<int> queue(1), retries(1), failed(std::vector<int>{0});
    Device<unsigned> count(std::vector<unsigned>{1}), retry_count(std::vector<unsigned>{0}), next(std::vector<unsigned>{0});
    Device<ContactOutput> outputs(sentinels);
    unsigned bad_cases = 0;
    for (bool parallel : {false,true}) for (std::size_t i = 0; i < records.size(); ++i) if (expected_error(small_reference[i].status)) {
        queue.upload({int(i)}); failed.upload({0}); next.upload({0}); retry_count.upload({0});
        launch_cooperative_exact_contacts(inputs.data, outputs.data, queue.data, count.data,
            retries.data, retry_count.data, next.data, words.data, slots.data, 1, failed.data,
            nullptr, nullptr, parallel);
        cuda_check(cudaGetLastError());
        require(failed.download()[0] == expected_error(small_reference[i].status), "bad-input error mapping");
        require(retry_count.download()[0] == 0, "bad-input must not become capacity retry");
        ++bad_cases;
    }
    require(bad_cases > 0, "error routing not exercised");
    Device<int> ownership(std::vector<int>{0,0});
    // Fixture zero is a finite active PT interior contact with friction.
    for (bool conversions : {false,true}) for (unsigned limit : {0u,1u,3u,8u,16u,64u,128u}) {
        ownership.upload({0,0});
        ownership_kernel<<<1,32>>>(inputs.data, words.data, slots.data, limit, ownership.data, conversions);
        cuda_check(cudaGetLastError());
        const auto result = ownership.download();
        require(result[0] == 0, "borrowed views modified source or corrupted ownership");
        if (!conversions && limit <= 8) require(result[1] == 8, "tiny entry helper pools must fail explicitly");
        if (conversions && limit <= 1) require(result[1] == 7, "tiny conversion helper pools must fail explicitly");
        if (limit >= 64) require(result[1] == 0, "normal helper pools must succeed");
    }
    require(cooperative_exact_arena_sizes(0).words == 0, "empty arena sizing");
    std::cout << "CUDA error routing=" << bad_cases << "; helper exhaustion/ownership PASS\n";
    // Separate suite: does not alter the existing record stream or its
    // summaries, but must run in the real test main.
    ee_policy_cases();
}
