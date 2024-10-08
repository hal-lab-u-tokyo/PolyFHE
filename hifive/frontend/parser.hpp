#pragma once

#include <memory>
#include <string>

#include "hifive/core/graph/graph.hpp"

namespace hifive {
namespace frontend {

std::shared_ptr<hifive::core::Graph> ParseDotToGraph(const std::string& dot);
} // namespace frontend
} // namespace hifive