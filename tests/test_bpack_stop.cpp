// ButterflyPACK adapter, GOTCHA(fortran-stop): with hopeless sampling options (rank0 4,
// rankrate 1, itermax 2: five samples per try for a level of rank about 150) ButterflyPACK
// ends the whole process with a Fortran STOP, exit status 0, in the middle of the
// construction. The test runs itself as a child process three times and checks what the
// parent sees:
//  1. default stop_exit_status: exit status 70, the adapter's diagnostic on stderr (cause and
//     options to change), ButterflyPACK's own message on stdout (flushed before the status
//     is replaced), and bpack_from_matvec never returned;
//  2. stop_exit_status = 0: ButterflyPACK's status 0, the same diagnostic;
//  3. a normal construction and exit: status 0 and no diagnostic.
#include "hmb_bpack.hpp"
#include "test_bpack_h2opus_common.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

using namespace hmb;
using namespace hmb::test;

namespace {

int child(const std::string &mode) {
    Problem P(1000);
    const BlackBox D = dense_kernel_blackbox(P);
    BpackOptions o;
    if (mode != "ok") {
        o.rank0 = 4;
        o.rankrate = 1.0;
        o.itermax = 2;
    }
    if (mode == "hopeless-status0") o.stop_exit_status = 0;
    std::printf("CHILD-STARTED %s\n", mode.c_str());
    std::fflush(stdout);
    BpackHandle h = bpack_from_matvec(D, 3, P.xyz.data(), o);
    std::printf("CHILD-RETURNED\n");
    return 0;
}

std::string slurp(const std::string &path) {
    std::ifstream f(path);
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}

struct Run {
    bool exited = false;
    int status = -1;
    std::string out, err;
    bool out_has(const char *s) const { return out.find(s) != std::string::npos; }
    bool err_has(const char *s) const { return err.find(s) != std::string::npos; }
};

Run run_child(const std::string &self, const std::string &dir, const std::string &mode) {
    const std::string out = dir + "/" + mode + ".out", err = dir + "/" + mode + ".err";
    // A STOP leaves ButterflyPACK's memory allocated: keep LeakSanitizer quiet in ASan builds.
    const std::string cmd = "ASAN_OPTIONS=\"${ASAN_OPTIONS:+$ASAN_OPTIONS:}detect_leaks=0\" '" + self + "' --child " +
                            mode + " > '" + out + "' 2> '" + err + "'";
    const int rc = std::system(cmd.c_str());
    Run r;
    if (rc != -1 && WIFEXITED(rc)) {
        r.exited = true;
        r.status = WEXITSTATUS(rc);
    }
    r.out = slurp(out);
    r.err = slurp(err);
    std::remove(out.c_str());
    std::remove(err.c_str());
    std::printf("---- child '%s': %s %d\n---- stdout:\n%s---- stderr:\n%s----\n", mode.c_str(),
                r.exited ? "exit status" : "abnormal end, raw status", r.exited ? r.status : rc, r.out.c_str(),
                r.err.c_str());
    return r;
}

} // namespace

int main(int argc, char **argv) {
    const char *mode = arg_str(argc, argv, "--child", nullptr);
    if (mode) return child(mode);

    Checker C("test_bpack_stop");
    char self[4096];
    const ssize_t len = readlink("/proc/self/exe", self, sizeof self - 1);
    if (len <= 0) {
        C.check(false, "find own executable (/proc/self/exe)");
        return C.finish();
    }
    self[len] = 0;
    const char *tmp = std::getenv("TMPDIR");
    std::string tmpl = std::string(tmp && *tmp ? tmp : "/tmp") + "/hmb_bpack_stop.XXXXXX";
    if (!mkdtemp(&tmpl[0])) {
        C.check(false, "create a temporary directory");
        return C.finish();
    }
    const char *diag = "hmatrix-bridges: the process is exiting while ButterflyPACK builds a HODLR matrix from matvecs";

    Run r = run_child(self, tmpl, "hopeless");
    C.check(r.exited && r.status == 70, "STOP: exit status replaced by stop_exit_status", "status %d, want 70", r.status);
    C.check(r.err_has(diag), "STOP: diagnostic on stderr");
    C.check(r.err_has("rank0") && r.err_has("rankrate") && r.err_has("itermax"), "STOP: diagnostic names the options");
    C.check(r.out_has("randomized scheme not converged"), "STOP: ButterflyPACK's message flushed to stdout");
    C.check(r.out_has("CHILD-STARTED") && !r.out_has("CHILD-RETURNED"), "STOP: construction never returned");

    r = run_child(self, tmpl, "hopeless-status0");
    C.check(r.exited && r.status == 0, "STOP with stop_exit_status = 0: ButterflyPACK's status 0 kept", "status %d",
            r.status);
    C.check(r.err_has(diag) && r.err_has("Keeping ButterflyPACK's exit status"), "STOP with status 0: diagnostic on stderr");
    C.check(r.out_has("randomized scheme not converged") && !r.out_has("CHILD-RETURNED"),
            "STOP with status 0: ButterflyPACK's message, no return");

    r = run_child(self, tmpl, "ok");
    C.check(r.exited && r.status == 0 && r.out_has("CHILD-RETURNED"), "normal construction returns", "status %d",
            r.status);
    C.check(!r.err_has("hmatrix-bridges:"), "no diagnostic after a normal construction and exit");

    rmdir(tmpl.c_str());
    return C.finish();
}
