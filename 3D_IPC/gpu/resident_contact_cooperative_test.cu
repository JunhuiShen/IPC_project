// nvcc -std=c++17 -O2 --fmad=false -I gpu \
//   gpu/resident_contact_cooperative_test.cu gpu/resident_contact_cooperative.cu \
//   -o /tmp/contact_cooperative_cuda_test
#define IPC_CONTACT_COOPERATIVE_CUDA_TEST
#include "resident_contact_cooperative_test.cpp"
#include "resident_contact_cooperative.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <new>
#include <numeric>

namespace {
namespace exact = ipc_gpu::resident_exact;
namespace coop = ipc_gpu::resident::contact_cooperative_detail;
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
}
