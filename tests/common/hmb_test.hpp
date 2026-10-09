// hmatrix-bridges test and example harness (not part of the library API).
//
// Test problems, timers, error measures, memory probes and the pass sentinel shared by
// tests/ and examples/. Header-only; includes no third-party library.
#pragma once
#include "hmb/blackbox.hpp"
#include "hmb/entry_source.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace hmb {
namespace test {

// ---------------------------------------------------------------------------------------
// Test problems: N points on the unit sphere (user ordering = generation order) and a
// smooth kernel. These are the problems behind the measurements in RESULTS.md.
//
//   Kernel::Sym     K(x, y) = 1 / (0.1 + |x - y|)                      (copy and matvec routes)
//   Kernel::Nonsym  K(x, y) = (1 + 0.5 (x_z - y_z)) / (4 pi sqrt(|x - y|^2 + h^2)),
//                   h^2 = 4 pi / N (mean spacing squared)              (entry route)
//
//   Geometry::RandomSphere       normalised Gaussian samples (seeded)
//   Geometry::ShuffledFibonacci  Fibonacci lattice, user numbering randomly shuffled
//
// shift is added to the diagonal (makes the matrix comfortably invertible for factorizations).
// ---------------------------------------------------------------------------------------
enum class Kernel { Sym, Nonsym };
enum class Geometry { RandomSphere, ShuffledFibonacci };

struct Problem {
    int n = 0;
    Kernel kernel = Kernel::Sym;
    double shift = 0;
    double h2 = 0;
    std::vector<double> xyz; // point-major: xyz[3 * i + d]

    Problem(int n_, Kernel k = Kernel::Sym, Geometry g = Geometry::RandomSphere, double shift_ = 0, unsigned seed = 0)
        : n(n_), kernel(k), shift(shift_), xyz(3 * size_t(n_)) {
        if (g == Geometry::RandomSphere) {
            std::mt19937 gen(seed);
            std::normal_distribution<double> nd;
            for (int i = 0; i < n; i++) {
                double v[3], r = 0;
                for (double &c : v) { c = nd(gen); r += c * c; }
                r = std::sqrt(r);
                for (int d = 0; d < 3; d++) xyz[3 * size_t(i) + d] = v[d] / r;
            }
        } else {
            std::vector<int> p(n);
            std::iota(p.begin(), p.end(), 0);
            std::mt19937 gen(seed ? seed : 12345);
            std::shuffle(p.begin(), p.end(), gen);
            const double ga = M_PI * (3. - std::sqrt(5.));
            for (int i = 0; i < n; i++) {
                double z = 1 - 2 * (i + 0.5) / n, r = std::sqrt(1 - z * z), ph = ga * i;
                xyz[3 * size_t(p[i])] = r * std::cos(ph);
                xyz[3 * size_t(p[i]) + 1] = r * std::sin(ph);
                xyz[3 * size_t(p[i]) + 2] = z;
            }
        }
        h2 = 4 * M_PI / n;
    }

    bool symmetric() const { return kernel == Kernel::Sym; }

    double K(int i, int j) const {
        const double *a = &xyz[3 * size_t(i)], *b = &xyz[3 * size_t(j)];
        double dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
        double v = kernel == Kernel::Sym ? 1.0 / (0.1 + std::sqrt(dx * dx + dy * dy + dz * dz))
                                         : (1.0 + 0.5 * dz) / (4 * M_PI * std::sqrt(dx * dx + dy * dy + dz * dz + h2));
        return i == j ? v + shift : v;
    }

    // Exact Y = op(K) X for k columns, O(n^2 k), original ordering, column-major (ld n).
    void exact_apply(char trans, int k, const double *X, double *Y) const {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            std::vector<double> s(k, 0.0);
            for (int j = 0; j < n; j++) {
                double kij = (trans == 'N' || trans == 'n') ? K(i, j) : K(j, i);
                for (int v = 0; v < k; v++) s[v] += kij * X[j + size_t(v) * n];
            }
            for (int v = 0; v < k; v++) Y[i + size_t(v) * n] = s[v];
        }
    }

    // Entry oracle for the kernel (thread-safe).
    EntrySource entries() const {
        EntrySource e;
        e.n_rows = e.n_cols = n;
        e.thread_safe = true;
        const Problem *self = this;
        e.get = [self](int M, int N, const int *rows, const int *cols, double *out) {
            for (int c = 0; c < N; c++)
                for (int r = 0; r < M; r++) out[r + size_t(M) * c] = self->K(rows[r], cols[c]);
        };
        return e;
    }

    // Exact operator as a BlackBox (O(n^2) per column; for reference values only).
    BlackBox exact_blackbox() const {
        BlackBox b;
        b.n = n;
        b.symmetric = symmetric();
        const Problem *self = this;
        b.apply = [self](char trans, int k, const double *X, double *Y) { self->exact_apply(trans, k, X, Y); };
        return b;
    }
};

// ---------------------------------------------------------------------------------------
// Measurement helpers
// ---------------------------------------------------------------------------------------
inline double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}
// ||a - ref||_F / ||ref||_F
inline double relerr(const std::vector<double> &a, const std::vector<double> &ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); i++) {
        num += (a[i] - ref[i]) * (a[i] - ref[i]);
        den += ref[i] * ref[i];
    }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}
inline std::vector<double> randvec(size_t m, unsigned seed) {
    std::mt19937 g(seed);
    std::normal_distribution<double> nd;
    std::vector<double> v(m);
    for (auto &t : v) t = nd(g);
    return v;
}
// Relative error of op(A) X against op(R) X for k random vectors (both BlackBoxes).
inline double blackbox_relerr(const BlackBox &A, const BlackBox &R, char trans = 'N', int k = 4, unsigned seed = 7) {
    std::vector<double> X = randvec(size_t(A.n) * k, seed), YA(size_t(A.n) * k), YR(size_t(A.n) * k);
    A.apply(trans, k, X.data(), YA.data());
    R.apply(trans, k, X.data(), YR.data());
    return relerr(YA, YR);
}

// /proc/self/status values in kB. reset_hwm() makes VmHWM restart from the current RSS.
inline long proc_kb(const char *key) {
    FILE *f = std::fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    long v = -1;
    size_t kl = std::strlen(key);
    while (std::fgets(line, sizeof line, f))
        if (!std::strncmp(line, key, kl) && line[kl] == ':') { v = std::atol(line + kl + 1); break; }
    std::fclose(f);
    return v;
}
inline void reset_hwm() {
    FILE *f = std::fopen("/proc/self/clear_refs", "w");
    if (f) { std::fputs("5", f); std::fclose(f); }
}
inline long rss_kb() { return proc_kb("VmRSS"); }
inline long hwm_kb() { return proc_kb("VmHWM"); }

inline int env_int(const char *k, int d) { const char *s = std::getenv(k); return s ? std::atoi(s) : d; }
inline double env_dbl(const char *k, double d) { const char *s = std::getenv(k); return s ? std::atof(s) : d; }

// Minimal "--name value" argument reader: arg_int(argc, argv, "--n", 2000).
inline const char *arg_str(int argc, char **argv, const char *name, const char *def) {
    for (int a = 1; a + 1 < argc; a++)
        if (!std::strcmp(argv[a], name)) return argv[a + 1];
    return def;
}
inline int arg_int(int argc, char **argv, const char *name, int def) {
    const char *s = arg_str(argc, argv, name, nullptr);
    return s ? std::atoi(s) : def;
}
inline double arg_dbl(int argc, char **argv, const char *name, double def) {
    const char *s = arg_str(argc, argv, name, nullptr);
    return s ? std::atof(s) : def;
}
inline bool arg_flag(int argc, char **argv, const char *name) {
    for (int a = 1; a < argc; a++)
        if (!std::strcmp(argv[a], name)) return true;
    return false;
}

// ---------------------------------------------------------------------------------------
// Checks. CHECK() records failures and keeps going; finish() prints the sentinel that
// tests/run_tests.sh requires (HMB_TEST_PASSED <summary>) only if nothing failed, and
// returns the process exit status.
// ---------------------------------------------------------------------------------------
struct Checker {
    std::string name;
    int failures = 0, checks = 0;
    explicit Checker(std::string n) : name(std::move(n)) { std::setvbuf(stdout, nullptr, _IOLBF, 0); }

    // Logs "check <what>: ok|FAILED" with an optional printf-style detail.
    bool check(bool ok, const char *what, const char *fmt = nullptr, ...) {
        checks++;
        char detail[512] = "";
        if (fmt) {
            va_list ap;
            va_start(ap, fmt);
            std::vsnprintf(detail, sizeof detail, fmt, ap);
            va_end(ap);
        }
        std::printf("check %-48s %s%s%s\n", what, ok ? "ok" : "FAILED", *detail ? "  " : "", detail);
        if (!ok) failures++;
        return ok;
    }
    // value <= bound, logged with both numbers
    bool check_le(double value, double bound, const char *what) {
        return check(value <= bound, what, "%.3e <= %.3e", value, bound);
    }
    int finish(const char *summary_fmt = nullptr, ...) {
        char summary[256] = "";
        if (summary_fmt) {
            va_list ap;
            va_start(ap, summary_fmt);
            std::vsnprintf(summary, sizeof summary, summary_fmt, ap);
            va_end(ap);
        }
        if (failures == 0) {
            std::printf("HMB_TEST_PASSED %s: %d checks%s%s\n", name.c_str(), checks, *summary ? "; " : "", summary);
            std::fflush(stdout);
            return 0;
        }
        std::printf("HMB_TEST_FAILED %s: %d of %d checks failed\n", name.c_str(), failures, checks);
        std::fflush(stdout);
        return 1;
    }
};

} // namespace test
} // namespace hmb
