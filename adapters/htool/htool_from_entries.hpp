// move-hmatrix: feed any EntrySource to Htool-DDM's compressor.
// Htool's VirtualGenerator uses the same convention as EntrySource (user ordering,
// 0-based, column-major M x N), so no index translation is needed.
#pragma once
#include "hmb/entry_source.hpp"
#include <htool/hmatrix/interfaces/virtual_generator.hpp>
#include <mutex>

namespace hmb {

class HtoolEntryGenerator : public htool::VirtualGenerator<double> {
    const EntrySource &m_src;
    mutable std::mutex m_lock; // serializes calls when the source is not thread-safe

  public:
    explicit HtoolEntryGenerator(const EntrySource &src) : m_src(src) {}

    void copy_submatrix(int M, int N, const int *rows, const int *cols, double *ptr) const override {
        if (m_src.thread_safe) {
            m_src.get(M, N, rows, cols, ptr);
            return;
        }
        std::lock_guard<std::mutex> guard(m_lock);
        m_src.get(M, N, rows, cols, ptr);
    }
};

} // namespace hmb
