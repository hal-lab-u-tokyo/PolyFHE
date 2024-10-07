#pragma once

#include <memory>
#include <string>

#include "hifive/core/graph/graph.hpp"

namespace hifive {
namespace frontend {

void ParseDotToGraph(const std::string& dot,
                     std::shared_ptr<hifive::core::Graph>& graph);
} // namespace frontend
} // namespace hifive