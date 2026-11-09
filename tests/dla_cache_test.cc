// Host unit test for neuron/npustt_dla_cache.h (added by patches/0004-dla-cache.patch).
#include "neuron/npustt_dla_cache.h"

#include <errno.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

using namespace tflite::neuron;

static int failures = 0;
static int checks = 0;
#define CHECK(cond)                                              \
  do {                                                           \
    ++checks;                                                    \
    if (!(cond)) {                                               \
      std::printf("FAIL line %d: %s\n", __LINE__, #cond);        \
      ++failures;                                                \
    }                                                            \
  } while (0)

static bool Exists(const std::string& p) {
  struct stat st;
  return stat(p.c_str(), &st) == 0;
}

static std::string ReadAll(const std::string& p) {
  std::string out;
  FILE* f = std::fopen(p.c_str(), "rb");
  if (f == nullptr) return out;
  char buf[4096];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
  std::fclose(f);
  return out;
}

static std::string Key(const std::vector<uint8_t>& tok, const char* acc,
                       const char* args, const char* opts, bool fp16,
                       const char* ver) {
  return NpusttDlaKey(tok, acc, args, opts, fp16, ver);
}

int main() {
  {  // mode selection
    CHECK(NpusttDlaRequested("/d", "t"));
    CHECK(!NpusttDlaRequested("/d", nullptr));
    CHECK(!NpusttDlaRequested("/d", ""));
    CHECK(!NpusttDlaRequested(nullptr, "t"));
    CHECK(!NpusttDlaRequested("", "t"));
    CHECK(NpusttDlaMissingSymbol(true, true, true) == nullptr);
    CHECK(std::strcmp(NpusttDlaMissingSymbol(false, true, true),
                      "NeuronCompilation_getCompiledNetworkSize") == 0);
    CHECK(std::strcmp(NpusttDlaMissingSymbol(true, false, true),
                      "NeuronCompilation_storeCompiledNetwork") == 0);
    CHECK(std::strcmp(NpusttDlaMissingSymbol(true, true, false),
                      "NeuronModel_restoreFromCompiledNetwork") == 0);
    CHECK(NpusttVersionString(8, 2, 26) == "8.2.26");
  }
  {  // FNV-1a 64 reference vectors
    NpusttFnv1a64 e;
    CHECK(e.value() == 0xcbf29ce484222325ULL);
    NpusttFnv1a64 a;
    a.Bytes("a", 1);
    CHECK(a.value() == 0xaf63dc4c8601ec8cULL);
    NpusttFnv1a64 fb;
    fb.Bytes("foobar", 6);
    CHECK(fb.value() == 0x85944171f73967e8ULL);
  }
  {  // key: format, determinism, every input counts, field boundaries count
    std::vector<uint8_t> tok(32);
    for (int i = 0; i < 32; ++i) tok[i] = static_cast<uint8_t>(i * 7);
    const std::string base =
        Key(tok, "mtk-mdla", "--opt-accuracy --opt 3", "--relax-fp32", true, "8.2.26");
    CHECK(base.size() == 16);
    CHECK(base.find_first_not_of("0123456789abcdef") == std::string::npos);
    CHECK(base == Key(tok, "mtk-mdla", "--opt-accuracy --opt 3", "--relax-fp32", true, "8.2.26"));
    std::vector<uint8_t> tok2 = tok;
    tok2[31] ^= 1;
    std::set<std::string> keys = {
        base,
        Key(tok2, "mtk-mdla", "--opt-accuracy --opt 3", "--relax-fp32", true, "8.2.26"),
        Key(tok, "mtk-dsp", "--opt-accuracy --opt 3", "--relax-fp32", true, "8.2.26"),
        Key(tok, "mtk-mdla", "--opt-accuracy --opt 2", "--relax-fp32", true, "8.2.26"),
        Key(tok, "mtk-mdla", "--opt-accuracy --opt 3", "", true, "8.2.26"),
        Key(tok, "mtk-mdla", "--opt-accuracy --opt 3", "--relax-fp32", false, "8.2.26"),
        Key(tok, "mtk-mdla", "--opt-accuracy --opt 3", "--relax-fp32", true, "8.2.27"),
        Key(tok, "", "--opt-accuracy --opt 3", "--relax-fp32", true, "8.2.26"),
        Key(tok, nullptr, "--opt-accuracy --opt 3", "--relax-fp32 ", true, "8.2.26"),
    };
    CHECK(keys.size() == 9);  // all distinct
    // "" and nullptr accelerator are the same input (no name set)
    CHECK(Key(tok, "", "x", "y", true, "v") == Key(tok, nullptr, "x", "y", true, "v"));
    // moving a byte between fields changes the key
    CHECK(Key(tok, "ab", "c", "", true, "v") != Key(tok, "a", "bc", "", true, "v"));
    std::printf("example key: %s\n", base.c_str());
  }
  {  // paths
    CHECK(NpusttDlaDir("/data/x/cache/neuron-dla", "0123abcdef") ==
          "/data/x/cache/neuron-dla/npustt-dla/0123abcdef");
    CHECK(NpusttDlaDir("/data/x/", "tok") == "/data/x/npustt-dla/tok");
    CHECK(NpusttDlaDir("/data/x//", "tok") == "/data/x/npustt-dla/tok");
    CHECK(NpusttDlaDir("/d", "../etc") == "/d/npustt-dla/.._etc");
    CHECK(NpusttDlaDir("/d", "..") == "/d/npustt-dla/_");
    CHECK(NpusttDlaDir("/d", "a b/c") == "/d/npustt-dla/a_b_c");
    CHECK(NpusttDlaDir("/d", "Model-1.v2_x") == "/d/npustt-dla/Model-1.v2_x");
    CHECK(NpusttDlaPath("/d/t", "00ff") == "/d/t/00ff.dla");
    CHECK(NpusttDlaPath("/d/t", "00ff", ".tmp") == "/d/t/00ff.tmp");
  }
  char tmpl[] = "/tmp/dla_cache_test_XXXXXX";
  const char* root_c = mkdtemp(tmpl);
  CHECK(root_c != nullptr);
  const std::string root = root_c;
  {  // mkdir -p
    const std::string deep = root + "/a/b/c";
    CHECK(NpusttMakeDirs(deep) == 0);
    CHECK(Exists(deep));
    CHECK(NpusttMakeDirs(deep) == 0);  // existing is fine
    FILE* f = std::fopen((root + "/file").c_str(), "w");
    std::fclose(f);
    CHECK(NpusttMakeDirs(root + "/file") == -ENOTDIR);
    CHECK(NpusttMakeDirs(root + "/file/sub") < 0);
  }
  {  // tmp-then-rename write, creates the directory
    const std::string dir = NpusttDlaDir(root.c_str(), "tok");
    std::string payload(3 * 1024 * 1024 + 17, '\0');
    for (size_t i = 0; i < payload.size(); ++i) payload[i] = static_cast<char>(i * 31 + 7);
    CHECK(NpusttWriteFileAtomic(dir, "k1", payload.data(), payload.size()) == 0);
    CHECK(ReadAll(NpusttDlaPath(dir, "k1")) == payload);
    CHECK(!Exists(NpusttDlaPath(dir, "k1", ".tmp")));
    // overwrite with something shorter
    CHECK(NpusttWriteFileAtomic(dir, "k1", "xyz", 3) == 0);
    CHECK(ReadAll(NpusttDlaPath(dir, "k1")) == "xyz");
    // directory cannot be created -> error, nothing left behind
    const std::string bad = root + "/file/sub";
    CHECK(NpusttWriteFileAtomic(bad, "k2", "abc", 3) < 0);
    CHECK(!Exists(bad + "/k2.dla") && !Exists(bad + "/k2.tmp"));
    // read-only directory -> -EACCES at open (skipped when running as root)
    if (geteuid() != 0) {
      const std::string ro = root + "/ro";
      CHECK(NpusttMakeDirs(ro) == 0);
      chmod(ro.c_str(), 0555);
      CHECK(NpusttWriteFileAtomic(ro, "k3", "abc", 3) == -EACCES);
      chmod(ro.c_str(), 0755);
    }
  }
  {  // mmap helper
    const std::string dir = NpusttDlaDir(root.c_str(), "tok");
    NpusttMappedFile m;
    CHECK(m.Map(NpusttDlaPath(dir, "missing")) == -ENOENT);
    CHECK(m.data() == nullptr && m.size() == 0);
    FILE* f = std::fopen((dir + "/empty.dla").c_str(), "w");
    std::fclose(f);
    CHECK(m.Map(dir + "/empty.dla") == -EINVAL);
    std::string payload(1024 * 1024 + 5, '\0');
    for (size_t i = 0; i < payload.size(); ++i) payload[i] = static_cast<char>(i * 13 + 1);
    CHECK(NpusttWriteFileAtomic(dir, "k4", payload.data(), payload.size()) == 0);
    CHECK(m.Map(NpusttDlaPath(dir, "k4")) == 0);
    CHECK(m.size() == payload.size());
    CHECK(std::memcmp(m.data(), payload.data(), payload.size()) == 0);
    m.ReleasePages();  // pages dropped, contents still readable from the file
    CHECK(std::memcmp(m.data(), payload.data(), payload.size()) == 0);
    // the mapping survives unlink of the file (as after a restore)
    unlink(NpusttDlaPath(dir, "k4").c_str());
    CHECK(std::memcmp(m.data(), payload.data(), payload.size()) == 0);
    NpusttMappedFile moved = std::move(m);
    CHECK(m.data() == nullptr && m.size() == 0);
    CHECK(moved.size() == payload.size());
    CHECK(static_cast<const char*>(moved.data())[100] == payload[100]);
    moved.Reset();
    CHECK(moved.data() == nullptr);
  }
  {  // clock
    const long long a = NpusttNowMs();
    usleep(20000);
    const long long b = NpusttNowMs();
    CHECK(b - a >= 19 && b - a < 1000);
  }
  const std::string rm = "rm -rf '" + root + "'";
  CHECK(std::system(rm.c_str()) == 0);
  std::printf(failures == 0 ? "ALL PASS (%d checks)\n" : "%d FAILURE(S)\n",
              failures == 0 ? checks : failures);
  return failures == 0 ? 0 : 1;
}
