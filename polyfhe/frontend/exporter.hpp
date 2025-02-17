#pragma once

#include <memory>
#include <string>

#include "polyfhe/core/graph/graph.hpp"
#include "polyfhe/frontend/parser.hpp"

namespace polyfhe {
namespace frontend {

void export_graph_to_dot(std::shared_ptr<polyfhe::core::Graph>& graph,
                         std::string filename);

} // namespace frontend
} // namespace polyfhe