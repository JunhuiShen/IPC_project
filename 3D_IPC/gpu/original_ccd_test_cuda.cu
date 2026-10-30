#include "original_ccd.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace orig = ipc_gpu::original_ccd;
namespace exact = ipc_gpu::exact;

namespace {

constexpr int kArenaWidth = 2048;
constexpr int kArenaSlots = 512;

// Every result record crosses the API boundary as a raw 16-byte object:
// cudaMemcpy is given `count * sizeof(orig::Result)`, so the WHOLE object
// representation of each Result -- not just its semantic fields -- is
// transferred. Result's three formerly-implicit padding bytes (offsets 5..7)
// are an explicit, zero-initialized member in original_ccd.cuh, so the
// fieldwise `results[i] = result` stores below define the complete record and
// the DtoH copy never reads indeterminate bytes. Do not replace the record
// store with a partial/per-field transfer.
static_assert(sizeof(orig::Result) == 16,
              "Result must be a 16-byte transfer record");

void check(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}

// The floating original backend is allocation-free and can run one worker per
// query. It has no exact-arithmetic context.
__global__ void original_fast_kernel(const orig::Query* queries,
                                     orig::Result* results,
                                     std::size_t count) {
    const std::size_t index =
        std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    // Whole-record store: `Result` has no implicit padding left, so this
    // writes all 16 transferred bytes (status, collision, reserved, toi).
    results[index] = orig::evaluate_fast(queries[index]);
}

__device__ void record_exact_status(orig::test::DeviceDiagnostics* diagnostics,
                                    const orig::Result& result) {
    ++diagnostics->exact_queries;
    if (result.status == orig::Status::Ok) ++diagnostics->exact_ok;
    else if (result.status == orig::Status::CapacityExceeded)
        ++diagnostics->exact_capacity_exceeded;
    else if (result.status == orig::Status::InvalidInput)
        ++diagnostics->exact_invalid;
    else ++diagnostics->exact_other;
}

__device__ void note_peak(orig::test::DeviceDiagnostics* diagnostics,
                          const exact::DevicePool& pool) {
    if (pool.peak > diagnostics->peak_slots)
        diagnostics->peak_slots = pool.peak;
}

// One cold exact query. The arena is installed before every evaluate call and
// released after the query. A 256->2048 retry is permitted ONLY after the
// first pass returned, all of its temporaries died, and the pool proved
// healthy (no failure flag, every slot returned). An allocator failure is
// never remapped to a mathematical retry.
__device__ orig::Result original_exact_one(
        const orig::Query& query, std::uint32_t* words, int* free_slots,
        orig::test::DeviceDiagnostics* diagnostics) {
    exact::DevicePool pool;
    exact::initialize_pool(pool, words, free_slots,
                           kArenaWidth, kArenaSlots);
    orig::Result result = orig::evaluate_exact<256>(query);

    // Capture ownership state BEFORE any reinitialization. evaluate_exact has
    // already converted an unhealthy pool to CapacityExceeded for every
    // non-InvalidInput query.
    const bool first_failed = pool.failed;
    const bool first_mismatch = pool.available != pool.slot_count;
    note_peak(diagnostics, pool);
    if (first_failed) ++diagnostics->first_pass_failed;
    if (first_mismatch) ++diagnostics->first_pass_available_mismatch;

    if (result.status == orig::Status::CapacityExceeded) {
        if (first_failed || first_mismatch) {
            // A corrupt/exhausted allocator is an explicit resource failure,
            // not evidence of limb-capacity exhaustion.
            ++diagnostics->retry_resource_errors;
        } else {
            // The first pass released every slot without failure, so this is
            // genuine limb-capacity exhaustion. Retry with the wider limb cap
            // from a clean arena; no arithmetic was truncated.
            ++diagnostics->retries_2048;
            exact::initialize_pool(pool, words, free_slots,
                                   kArenaWidth, kArenaSlots);
            result = orig::evaluate_exact<2048>(query);
            const bool final_failed = pool.failed;
            const bool final_mismatch = pool.available != pool.slot_count;
            note_peak(diagnostics, pool);
            if (final_failed) ++diagnostics->final_pass_failed;
            if (final_mismatch)
                ++diagnostics->final_pass_available_mismatch;
            if (final_failed || final_mismatch) {
                ++diagnostics->final_pass_errors;
                result = orig::Result{
                    orig::Status::CapacityExceeded, false, 1.0};
            }
        }
    } else if ((first_failed || first_mismatch)
               && result.status != orig::Status::InvalidInput) {
        // evaluate_exact must have produced CapacityExceeded in this case.
        // Treat any other outcome as a hidden ownership error.
        ++diagnostics->ownership_errors;
        result = orig::Result{orig::Status::CapacityExceeded, false, 1.0};
    }

    exact::release_pool();
    return result;
}

// Separate probe for the limb-capacity retry contract: capacity 2 is too small
// for an ordinary crossing's distance polynomial but must leave the arena
// healthy; the same query then resolves from a clean capacity-2048 pool. This
// verifies limb exhaustion independently of DevicePool exhaustion.
__device__ void small_capacity_probe(
        const orig::Query& query, std::uint32_t* words, int* free_slots,
        orig::test::DeviceDiagnostics* diagnostics) {
    exact::DevicePool pool;
    exact::initialize_pool(pool, words, free_slots,
                           kArenaWidth, kArenaSlots);
    const orig::Result narrow = orig::evaluate_exact<2>(query);
    if (narrow.status == orig::Status::CapacityExceeded)
        ++diagnostics->small_capacity_limb_failures;
    if (pool.failed) ++diagnostics->small_capacity_pool_failures;
    if (pool.available != pool.slot_count)
        ++diagnostics->small_capacity_available_mismatch;

    exact::initialize_pool(pool, words, free_slots,
                           kArenaWidth, kArenaSlots);
    const orig::Result wide = orig::evaluate_exact<2048>(query);
    if (wide.status == orig::Status::Ok && wide.collision)
        ++diagnostics->small_capacity_retry_ok;
    else
        ++diagnostics->small_capacity_retry_failures;
    if (pool.failed) ++diagnostics->small_capacity_retry_pool_failures;
    if (pool.available != pool.slot_count)
        ++diagnostics->small_capacity_retry_available_mismatch;
    exact::release_pool();
}

__global__ void original_exact_kernel(
        const orig::Query* queries, orig::Result* results, std::size_t count,
        const orig::Query* small_capacity_query,
        std::uint32_t* words, int* free_slots,
        orig::test::DeviceDiagnostics* diagnostics) {
    // A single cold worker mutates one arena. Fast queries already ran in
    // parallel and their results remain on device.
    if (blockIdx.x || threadIdx.x) return;
    diagnostics->slot_count = kArenaSlots;
    for (std::size_t i = 0; i < count; ++i) {
        const orig::Result result = original_exact_one(
            queries[i], words, free_slots, diagnostics);
        record_exact_status(diagnostics, result);
        // Whole-record store, as above: all 16 transferred bytes are defined.
        results[i] = result;
    }
    if (small_capacity_query)
        small_capacity_probe(*small_capacity_query, words, free_slots,
                             diagnostics);
}

template <class T>
struct DeviceBuffer {
    T* pointer = nullptr;
    explicit DeviceBuffer(std::size_t count) {
        if (count)
            check(cudaMalloc(reinterpret_cast<void**>(&pointer),
                             count * sizeof(T)));
    }
    ~DeviceBuffer() { if (pointer) cudaFree(pointer); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

} // namespace

void original_ccd_cuda_evaluate(const orig::Query* queries,
                                std::size_t count,
                                orig::Result* fast,
                                orig::Result* exact_results,
                                const orig::Query* small_capacity_query,
                                orig::test::DeviceDiagnostics* diagnostics) {
    if (!count) return;
    DeviceBuffer<orig::Query> device_queries(count);
    DeviceBuffer<orig::Result> device_fast(count);
    DeviceBuffer<orig::Result> device_exact(count);
    DeviceBuffer<std::uint32_t> device_words(kArenaWidth * kArenaSlots);
    DeviceBuffer<int> device_slots(kArenaSlots);
    DeviceBuffer<orig::Query> device_small_query(small_capacity_query ? 1 : 0);
    DeviceBuffer<orig::test::DeviceDiagnostics> device_diagnostics(1);

    check(cudaMemset(device_diagnostics.pointer, 0,
                     sizeof(orig::test::DeviceDiagnostics)));
    check(cudaMemcpy(device_queries.pointer, queries,
                     count * sizeof(orig::Query), cudaMemcpyHostToDevice));
    if (small_capacity_query)
        check(cudaMemcpy(device_small_query.pointer, small_capacity_query,
                         sizeof(orig::Query), cudaMemcpyHostToDevice));

    const unsigned threads = 128;
    const unsigned blocks =
        static_cast<unsigned>((count + threads - 1) / threads);
    original_fast_kernel<<<blocks, threads>>>(
        device_queries.pointer, device_fast.pointer, count);
    check(cudaGetLastError());
    original_exact_kernel<<<1, 1>>>(
        device_queries.pointer, device_exact.pointer, count,
        small_capacity_query ? device_small_query.pointer : nullptr,
        device_words.pointer, device_slots.pointer,
        device_diagnostics.pointer);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(fast, device_fast.pointer,
                     count * sizeof(orig::Result),
                     cudaMemcpyDeviceToHost));
    check(cudaMemcpy(exact_results, device_exact.pointer,
                     count * sizeof(orig::Result),
                     cudaMemcpyDeviceToHost));
    if (diagnostics)
        check(cudaMemcpy(diagnostics, device_diagnostics.pointer,
                         sizeof(orig::test::DeviceDiagnostics),
                         cudaMemcpyDeviceToHost));
}
