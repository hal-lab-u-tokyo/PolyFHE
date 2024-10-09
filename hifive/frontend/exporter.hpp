#pragma once

#include <memory>
#include <string>

#include "hifive/core/graph/graph.hpp"
#include "hifive/frontend/parser.hpp"

namespace hifive {
namespace frontend {

void export_graph_to_dot(std::shared_ptr<hifive::core::Graph>& graph,
                         std::string filename);

} // namespace frontend
} // namespace hifive