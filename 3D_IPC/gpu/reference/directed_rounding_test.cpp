// Standalone host qualification for the host/device rounding helper. This
// does not execute or substitute for a CUDA-device test.
// Build with the gpu_directed_rounding_test CMake target (Eigen headers required).
#include "directed_rounding.cuh"
#include "../../DAT.h"

#include <array>
#include <cerrno>
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#if defined(__clang__)
#pragma STDC FENV_ACCESS ON
#endif

namespace {

std::size_t result_checks = 0, environment_checks = 0;

std::uint64_t bits(double x) {
    std::uint64_t pattern;
    std::memcpy(&pattern, &x, sizeof(pattern));
    return pattern;
}
double value(std::uint64_t pattern) {
    double x;
    std::memcpy(&x, &pattern, sizeof(x));
    return x;
}

// Force the oracle to call platform libm rather than fold special inputs.
double (*volatile platform_nextafter)(double, double) = std::nextafter;
double reference_up(double x) { return platform_nextafter(x, INFINITY); }
double reference_down(double x) { return platform_nextafter(x, -INFINITY); }
double reference_zero(double x) { return platform_nextafter(x, 0.0); }

using Unary = double (*)(double);
struct Operation { const char* name; Unary reference, cpu, gpu; };
const std::array<Operation, 3> operations{{
    {"next_up", reference_up, directed_rounding::next_up, ipc_gpu::directed_rounding::next_up},
    {"next_down", reference_down, directed_rounding::next_down, ipc_gpu::directed_rounding::next_down},
    {"toward_positive_zero", reference_zero, directed_rounding::toward_positive_zero,
        ipc_gpu::directed_rounding::toward_positive_zero}
}};

class ScopedEnvironment {
    std::fenv_t environment_;
    int saved_errno_ = errno;
    bool saved_;
public:
    ScopedEnvironment() : saved_(std::fegetenv(&environment_) == 0) {
        if (!saved_) throw std::runtime_error("Cannot save floating-point environment");
    }
    ~ScopedEnvironment() {
        if (saved_) std::fesetenv(&environment_);
        errno = saved_errno_;
    }
};

struct Observation { std::uint64_t output; int error, exceptions; };
Observation observe(Unary operation, std::uint64_t input, int initial_errno, int initial_exceptions) {
    std::feclearexcept(FE_ALL_EXCEPT);
    if (initial_exceptions) std::feraiseexcept(initial_exceptions);
    errno = initial_errno;
    const auto output = bits(operation(value(input)));
    const int error = errno;
    const int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
    return {output, error, exceptions};
}

void check_bits(const Operation& operation, std::uint64_t input) {
    const auto expected = bits(operation.reference(value(input)));
    for (auto actual : {bits(operation.cpu(value(input))), bits(operation.gpu(value(input)))}) {
        ++result_checks;
        if (actual != expected) {
            std::ostringstream message;
            message << operation.name << " input=0x" << std::hex << input
                << " expected=0x" << expected << " actual=0x" << actual;
            throw std::runtime_error(message.str());
        }
    }
}

void check_environment(const Operation& operation, std::uint64_t input,
    int initial_errno, int initial_exceptions) {
    const auto expected = observe(operation.reference, input, initial_errno, initial_exceptions);
    for (auto implementation : {operation.cpu, operation.gpu}) {
        const auto actual = observe(implementation, input, initial_errno, initial_exceptions);
        ++environment_checks;
        if (actual.output != expected.output || actual.error != expected.error
            || actual.exceptions != expected.exceptions) {
            std::ostringstream message;
            message << operation.name << " input=0x" << std::hex << input
                << " output expected=0x" << expected.output << " actual=0x" << actual.output
                << std::dec << " errno expected=" << expected.error << " actual=" << actual.error
                << " exceptions expected=" << expected.exceptions << " actual=" << actual.exceptions;
            throw std::runtime_error(message.str());
        }
    }
}

std::uint64_t next_pattern(std::uint64_t& state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

void signed_zeros() {
    for (auto zero : {UINT64_C(0), UINT64_C(0x8000000000000000)}) {
        for (const auto& operation : operations) check_bits(operation, zero);
        if (bits(ipc_gpu::directed_rounding::next_up(value(zero))) != UINT64_C(1)
            || bits(ipc_gpu::directed_rounding::next_down(value(zero))) != UINT64_C(0x8000000000000001)
            || bits(ipc_gpu::directed_rounding::toward_positive_zero(value(zero))) != UINT64_C(0))
            throw std::runtime_error("Signed-zero semantics changed");
    }
}

void qualify() {
    ScopedEnvironment restore;
    signed_zeros();
    constexpr std::array<std::uint64_t, 18> magnitudes{{
        0, 1, 2, 3, UINT64_C(0x000ffffffffffffe), UINT64_C(0x000fffffffffffff),
        UINT64_C(0x0010000000000000), UINT64_C(0x0010000000000001),
        UINT64_C(0x3ff0000000000000), UINT64_C(0x3ff0000000000001),
        UINT64_C(0x7feffffffffffffe), UINT64_C(0x7fefffffffffffff),
        UINT64_C(0x7ff0000000000000), UINT64_C(0x7ff0000000000001),
        UINT64_C(0x7ff0000000001234), UINT64_C(0x7ff7ffffffffffff),
        UINT64_C(0x7ff8000000000000), UINT64_C(0x7fffffffffffffff)
    }};
    constexpr std::array<int, 4> modes{{FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}};
    std::uint64_t state = UINT64_C(0x9a8c00f27487c1e3);
    for (int mode : modes) {
        if (std::fesetround(mode)) throw std::runtime_error("Unsupported rounding mode");
        for (auto magnitude : magnitudes)
            for (auto sign : {UINT64_C(0), UINT64_C(0x8000000000000000)})
                for (const auto& operation : operations)
                    for (int initial_errno : {0, 123})
                        for (int initial_exceptions : {0, FE_DIVBYZERO})
                            check_environment(operation, magnitude | sign, initial_errno, initial_exceptions);
        for (int sample = 0; sample < 4096; ++sample) {
            const auto input = next_pattern(state);
            for (const auto& operation : operations) check_environment(operation, input, 123, FE_DIVBYZERO);
        }
    }
    if (std::fesetround(FE_TONEAREST)) throw std::runtime_error("Cannot restore nearest rounding");
    for (std::uint64_t exponent = 1; exponent < 2047; ++exponent)
        for (auto mantissa : {UINT64_C(0), UINT64_C(1),
                UINT64_C(0x000ffffffffffffe), UINT64_C(0x000fffffffffffff)})
            for (auto sign : {UINT64_C(0), UINT64_C(0x8000000000000000)})
                for (const auto& operation : operations)
                    check_bits(operation, sign | (exponent << 52) | mantissa);
    for (int sample = 0; sample < 250000; ++sample) {
        const auto input = next_pattern(state);
        for (const auto& operation : operations) check_bits(operation, input);
    }
}

} // namespace

int main() {
    try {
        qualify();
        std::cout << "GPU directed-rounding HOST checks passed: " << result_checks
            << " output comparisons, " << environment_checks
            << " errno/fenv comparisons across four rounding modes. No CUDA execution.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "GPU directed-rounding HOST check failed: " << error.what() << '\n';
        return 1;
    }
}
