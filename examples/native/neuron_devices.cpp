// Lists the Neuron runtime version and devices of the phone, without the delegate: which adapter library
// opens, its version, and the device names to pass to NeuronDelegate.Options.setAcceleratorName().
//
// Build it into your app's own JNI library (NDK, C++17, link with -ldl) and log the lines. The app needs
//   <uses-native-library android:name="libneuronusdk_adapter.mtk.so" android:required="false"/>
// in its manifest (API >= 31), or dlopen() of the vendor library fails.
//
// Example output on a Dimensity 1080 (MT6877):
//   try	libneuronusdk_adapter.mtk.so	ok
//   lib	/system_ext/lib64/libneuronusdk_adapter.mtk.so
//   version	8.2.26
//   count	3
//   device	mtk-gpu	GPU
//   device	mtk-dsp	VPU
//   device	mtk-mdla	MDLA_2_0
//
// The function signatures follow MediaTek's MIT neuron_types.h in the delegate and ExecuTorch's
// NeuronAdapterShim.h (BSD).
#include "neuron_devices.h"

#include <dlfcn.h>

#include <cstdint>
#include <string>

namespace {
typedef struct {
    uint8_t major, minor, patch;
} NeuronRuntimeVersion;
typedef struct NeuronDevice NeuronDevice;

typedef int (*Fn_getVersion)(NeuronRuntimeVersion*);
typedef int (*Fn_getDeviceCount)(uint32_t*);
typedef int (*Fn_getDevice)(uint32_t, NeuronDevice**);
typedef int (*Fn_getName)(const NeuronDevice*, const char**);
typedef int (*Fn_getDescription)(const NeuronDevice*, const char**);

const char* const kCandidates[] = {
    "libneuronusdk_adapter.mtk.so",
    "libneuronusdk_adapter.mtk.so.5",
    "libneuronusdk_adapter.7.mtk.so",
    "libneuronusdk_adapter.6.mtk.so",
    "libneuronusdk_adapter.5.mtk.so",
};

std::string Err() {
    const char* e = dlerror();
    return e ? e : "(no dlerror)";
}
}  // namespace

namespace neuron_bridge {

std::vector<std::string> NeuronInfo() {
    std::vector<std::string> out;
    void* h = nullptr;
    for (const char* name : kCandidates) {
        dlerror();
        h = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
        if (h) {
            out.push_back(std::string("try\t") + name + "\tok");
            break;
        }
        out.push_back(std::string("try\t") + name + "\t" + Err());
    }
    if (!h) {
        out.push_back("ERROR\tno Neuron adapter library could be opened (see try lines)");
        return out;
    }

    auto getVersion = reinterpret_cast<Fn_getVersion>(dlsym(h, "Neuron_getVersion"));
    auto getDeviceCount = reinterpret_cast<Fn_getDeviceCount>(dlsym(h, "Neuron_getDeviceCount"));
    auto getDevice = reinterpret_cast<Fn_getDevice>(dlsym(h, "Neuron_getDevice"));
    auto getName = reinterpret_cast<Fn_getName>(dlsym(h, "NeuronDevice_getName"));
    auto getDescription = reinterpret_cast<Fn_getDescription>(dlsym(h, "NeuronDevice_getDescription"));

    Dl_info info{};
    if (getVersion && dladdr(reinterpret_cast<void*>(getVersion), &info) && info.dli_fname) {
        out.push_back(std::string("lib\t") + info.dli_fname);
    } else {
        out.push_back("lib\t-");
    }

    if (!getVersion || !getDeviceCount || !getDevice || !getName || !getDescription) {
        std::string missing;
        if (!getVersion) missing += " Neuron_getVersion";
        if (!getDeviceCount) missing += " Neuron_getDeviceCount";
        if (!getDevice) missing += " Neuron_getDevice";
        if (!getName) missing += " NeuronDevice_getName";
        if (!getDescription) missing += " NeuronDevice_getDescription";
        out.push_back("ERROR\tmissing symbols:" + missing);
        return out;  // the handle stays open for the process lifetime
    }

    NeuronRuntimeVersion v{};
    int rc = getVersion(&v);
    if (rc == 0) {
        out.push_back("version\t" + std::to_string(v.major) + "." + std::to_string(v.minor) + "." +
                      std::to_string(v.patch));
    } else {
        out.push_back("version\terror " + std::to_string(rc));
    }

    uint32_t count = 0;
    rc = getDeviceCount(&count);
    if (rc != 0) {
        out.push_back("ERROR\tNeuron_getDeviceCount=" + std::to_string(rc));
        return out;
    }
    out.push_back("count\t" + std::to_string(count));
    for (uint32_t i = 0; i < count; ++i) {
        NeuronDevice* dev = nullptr;
        rc = getDevice(i, &dev);
        if (rc != 0 || dev == nullptr) {
            out.push_back("device\t?\tNeuron_getDevice(" + std::to_string(i) + ")=" + std::to_string(rc));
            continue;
        }
        const char* name = nullptr;
        const char* desc = nullptr;
        int rn = getName(dev, &name);
        int rd = getDescription(dev, &desc);
        std::string n = (rn == 0 && name) ? name : "?";
        std::string d = (rd == 0 && desc) ? desc : "getDescription=" + std::to_string(rd);
        out.push_back("device\t" + n + "\t" + d);
    }
    return out;
}

}  // namespace neuron_bridge
