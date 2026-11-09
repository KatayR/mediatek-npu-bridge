#pragma once
#include <string>
#include <vector>

namespace neuron_bridge {
// Lines: "try\t<name>\t<ok|dlerror text>" per dlopen attempt, then on success
// "lib\t<path>", "version\tM.m.p", "count\tN", "device\t<name>\t<description>"...;
// when no library opens or a symbol is missing: "ERROR\t<text>".
std::vector<std::string> NeuronInfo();
}  // namespace neuron_bridge
