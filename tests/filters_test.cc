// Host unit test for neuron/npustt_filters.h (added by patches/0003-op-exclusion-filters.patch).
#include "neuron/npustt_filters.h"

#include <cstdio>
#include <cstring>

#include "tensorflow/lite/builtin_ops.h"

using tflite::neuron::ExtractNpusttFilters;
using tflite::neuron::NpusttFilters;

static int failures = 0;
#define CHECK(cond)                                              \
  do {                                                           \
    if (!(cond)) {                                               \
      std::printf("FAIL line %d: %s\n", __LINE__, #cond);        \
      ++failures;                                                \
    }                                                            \
  } while (0)

static NeuronDelegateOptions Opts(const char* compile) {
  NeuronDelegateOptions o{};
  std::snprintf(o.compile_options, sizeof(o.compile_options), "%s", compile);
  return o;
}

int main() {
  {  // no token: string untouched (even odd spacing), no filters
    NpusttFilters f;
    auto out = ExtractNpusttFilters(Opts("--relax-fp32   --opt 3"), &f);
    CHECK(std::strcmp(out.compile_options, "--relax-fp32   --opt 3") == 0);
    CHECK(f.excluded.empty() && f.max_softmax == 0);
  }
  {  // exclude list stripped, remainder kept in order
    NpusttFilters f;
    auto out = ExtractNpusttFilters(
        Opts("--relax-fp32 --npustt-exclude=SOFTMAX,batch_matmul,GELU --opt 3"), &f);
    CHECK(std::strcmp(out.compile_options, "--relax-fp32 --opt 3") == 0);
    CHECK(f.excluded.size() == 3);
    CHECK(f.excluded.count(kTfLiteBuiltinSoftmax) == 1);
    CHECK(f.excluded.count(kTfLiteBuiltinBatchMatmul) == 1);
    CHECK(f.excluded.count(kTfLiteBuiltinGelu) == 1);
    CHECK(f.excluded_names == "SOFTMAX,BATCH_MATMUL,GELU");
  }
  {  // only tokens -> empty compile string; unknown name ignored; repeated token merges
    NpusttFilters f;
    auto out = ExtractNpusttFilters(
        Opts("--npustt-exclude=NOPE,MEAN --npustt-exclude=MEAN,FULLY_CONNECTED --npustt-max-softmax=1000000"), &f);
    CHECK(out.compile_options[0] == '\0');
    CHECK(f.excluded.size() == 2);
    CHECK(f.excluded.count(kTfLiteBuiltinMean) == 1);
    CHECK(f.excluded.count(kTfLiteBuiltinFullyConnected) == 1);
    CHECK(f.max_softmax == 1000000);
    CHECK(f.excluded_names == "MEAN,FULLY_CONNECTED");
  }
  {  // element count helper
    int raw[] = {3, 16, 499, 499};
    CHECK(tflite::neuron::ElementCount(reinterpret_cast<TfLiteIntArray*>(raw)) == 16LL * 499 * 499);
  }
  std::printf(failures == 0 ? "ALL PASS\n" : "%d FAILURE(S)\n", failures);
  return failures == 0 ? 0 : 1;
}
