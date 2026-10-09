// Helpers shared by the ButterflyPACK and H2Opus tests (tests/test_bpack_*.cpp,
// tests/test_h2opus_hara*.cpp): an exact dense operator for the test kernels, and a BlackBox
// wrapper that counts what a destination asks for. Includes no third-party header and
// declares no BLAS prototype (those of Htool and H2Opus clash, GOTCHA(header-clash)): the
// dense operator lives in test_bpack_h2opus_common.cpp.
#pragma once
#include "hmb_test.hpp"

#include <atomic>
#include <memory>
#include <stdexcept>

namespace hmb {
namespace test {

// The kernel matrix of a Problem stored densely (8 n^2 bytes): exact to round-off, and fast
// enough through BLAS to serve the thousands of sampling vectors of a randomized
// construction (Problem::exact_blackbox recomputes every entry for every vector).
// symmetric follows the kernel; 'T' multiplies by the transpose.
BlackBox dense_kernel_blackbox(const Problem &P);

// Counts the calls a destination makes into a BlackBox (thread-safe), and the largest number
// of calls in flight at once.
struct CallCounts {
    std::atomic<long> calls{0}, calls_transposed{0};
    std::atomic<long long> vectors{0};
    std::atomic<int> in_flight{0}, max_in_flight{0};
};

// The wrapped BlackBox keeps A's apply function (copied) and the counters alive. If
// throw_after > 0, call number throw_after (1-based) throws std::runtime_error instead.
inline BlackBox counted(const BlackBox &A, std::shared_ptr<CallCounts> c, long throw_after = 0) {
    BlackBox b = A;
    auto inner = A.apply;
    b.apply = [inner, c, throw_after](char trans, int k, const double *X, double *Y) {
        const long call = ++c->calls;
        if (trans == 'T' || trans == 't') c->calls_transposed++;
        c->vectors += k;
        const int f = ++c->in_flight;
        int m = c->max_in_flight.load();
        while (f > m && !c->max_in_flight.compare_exchange_weak(m, f)) {
        }
        struct Leave {
            CallCounts &c;
            ~Leave() { c.in_flight--; }
        } leave{*c};
        if (throw_after > 0 && call == throw_after) throw std::runtime_error("deliberate failure in the source");
        inner(trans, k, X, Y);
    };
    return b;
}

} // namespace test
} // namespace hmb
