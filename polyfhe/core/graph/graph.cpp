#include "polyfhe/core/graph/graph.hpp"

#include <iostream>
#include <sstream>

#include "graph.hpp"
#include "polyfhe/core/logger.hpp"
namespace polyfhe {
namespace core {

std::string to_string(SubgraphType subgraph_type) {
    switch (subgraph_type) {
    case SubgraphType::Elem:
        return "Elem";
    case SubgraphType::ElemLimb1:
        return "ElemLimb1";
    case SubgraphType::ElemLimb2:
        return "ElemLimb2";
    case SubgraphType::ElemSlot:
        return "ElemSlot";
    case SubgraphType::ElemLimb1Slot:
        return "ElemLimb1Slot";
    case SubgraphType::ElemLimb2Slot:
        return "ElemLimb2Slot";
    case SubgraphType::NoAccess:
        return "NoAccess";
    default:
        LOG_ERROR("Invalid SubgraphType\n");
        exit(1);
    }
}

std::ostream &operator<<(std::ostream &os, const SubgraphType &subgraph_type) {
    switch (subgraph_type) {
    case SubgraphType::Elem:
        os << "Elem";
        break;
    case SubgraphType::ElemLimb1:
        os << "ElemLimb1";
        break;
    case SubgraphType::ElemLimb2:
        os << "ElemLimb2";
        break;
    case SubgraphType::ElemSlot:
        os << "ElemSlot";
        break;
    case SubgraphType::ElemLimb1Slot:
        os << "ElemLimb1Slot";
        break;
    case SubgraphType::ElemLimb2Slot:
        os << "ElemLimb2Slot";
        break;
    case SubgraphType::NoAccess:
        os << "NoAccess";
        break;
    default:
        LOG_ERROR("Invalid SubgraphType\n");
        exit(1);
    }
    return os;
}

int SubGraph::get_max_limb() {
    int max = 0;
    for (auto node : m_nodes) {
        for (auto edge : node->get_in_edges()) {
            auto found =
                std::find(m_nodes.begin(), m_nodes.end(), edge->get_src());
            if (found != m_nodes.end()) {
                if (edge->get_limb() > max) {
                    max = edge->get_limb();
                }
            }
        }
        for (auto edge : node->get_out_edges()) {
            auto found =
                std::find(m_nodes.begin(), m_nodes.end(), edge->get_dst());
            if (found != m_nodes.end()) {
                if (edge->get_limb() > max) {
                    max = edge->get_limb();
                }
            }
        }
    }
    return max;
}

void Graph::add_edge(std::shared_ptr<Node> src, std::shared_ptr<Node> dst) {
    add_edge(src, dst, "");
}

void Graph::add_edge(std::shared_ptr<Node> src, std::shared_ptr<Node> dst,
                     std::string label) {
    if (src == nullptr || dst == nullptr) {
        LOG_ERROR("src or dst is nullptr\n");
        exit(1);
    }

    std::shared_ptr<Edge> edge = std::make_shared<Edge>(src, dst);
    src->add_outgoing(edge);
    dst->add_incoming(edge);
    edge->update_name();

    // TODO: more nice way to specify init edge from user
    std::vector<std::string> vec;
    std::stringstream ss(label);
    std::string token;
    while (std::getline(ss, token, '_')) {
        vec.push_back(token);
    }

    if (dst->get_op_type() == core::OpType::Malloc) {
        // label is {limb}_malloc_{n_poly}
        if (vec.size() != 3) {
            LOG_ERROR("Invalid label: %s\n", label.c_str());
            exit(1);
        }
        const int limb = std::stoi(vec[0]);
        const int n_poly = std::stoi(vec[2]);
        edge->set_limb(limb);
        edge->set_start_limb(0);
        edge->set_end_limb(limb);
        dst->set_malloc_limb(limb);
        dst->set_malloc_num_poly(n_poly);
    } else if (src->get_op_type() == core::OpType::Init ||
               dst->get_op_type() == core::OpType::End) {
        // label is {limb}_{init/end}_{idx}_{offset}
        if (vec.size() != 4) {
            LOG_ERROR("Invalid label: %s\n", label.c_str());
            exit(1);
        }
        const int limb = std::stoi(vec[0]);
        const int idx = std::stoi(vec[2]);
        const int offset = std::stoi(vec[3]);
        edge->set_limb(limb);
        edge->set_start_limb(0);
        edge->set_end_limb(limb);
        edge->set_idx_argc(idx);
        edge->set_offset(offset);
    } else if (!label.empty()) {
        // label is {limb}
        if (vec.size() != 1) {
            LOG_ERROR("Invalid label: %s, src node: %s\n", label.c_str(),
                      src->get_op_name().c_str());
            exit(1);
        }
        const int limb = std::stoi(vec[0]);
        edge->set_limb(limb);
        edge->set_start_limb(0);
        edge->set_end_limb(limb);
    }
}

void Graph::add_node(std::shared_ptr<Node> node) {
    int id = m_nodes.size();
    m_nodes.push_back(node);
    node->set_id(id);
}

void Graph::remove_node(std::shared_ptr<Node> node) {
    for (auto it = m_nodes.begin(); it != m_nodes.end(); it++) {
        if (*it == node) {
            // Since node_id is the index of the node in m_nodes,
            // we don't remove() the node from m_nodes to avoid
            // updating the node_id of other nodes.
            *it = nullptr;
            break;
        }
    }
}

std::shared_ptr<Edge> get_edge(std::shared_ptr<Node> src,
                               std::shared_ptr<Node> dst) {
    for (auto edge : src->get_out_edges()) {
        if (edge->get_dst() == dst) {
            return edge;
        }
    }
    return nullptr;
}

core::SubgraphType GetSubgraphType(
    std::vector<std::shared_ptr<polyfhe::core::Node>> &s) {
    if (s.size() == 0) {
        LOG_ERROR("Subgraph is empty\n");
        assert(false);
    }

    bool contains_elemwise = false;
    bool contains_limbwise = false;
    bool contains_ntt1 = false;
    bool contains_ntt2 = false;
    bool contains_slotwise = false;
    bool only_not_defined = true;
    for (auto node : s) {
        core::MemoryAccessPattern pattern = node->get_access_pattern();
        if (pattern != core::MemoryAccessPattern::NoAccess) {
            only_not_defined = false;
            break;
        }
    }
    for (auto node : s) {
        core::MemoryAccessPattern pattern = node->get_access_pattern();
        if (pattern == core::MemoryAccessPattern::ElementWise) {
            contains_elemwise = true;
        } else if (pattern == core::MemoryAccessPattern::LimbWise) {
            contains_limbwise = true;
            core::OpType op_type = node->get_op_type();
            if (op_type == core::OpType::NTTPhase1 ||
                op_type == core::OpType::iNTTPhase1) {
                contains_ntt1 = true;
            } else if (op_type == core::OpType::NTTPhase2 ||
                       op_type == core::OpType::iNTTPhase2) {
                contains_ntt2 = true;
            }
        } else if (pattern == core::MemoryAccessPattern::SlotWise) {
            contains_slotwise = true;
        } else if (pattern == core::MemoryAccessPattern::NoAccess) {
            // Do nothing
        } else {
            LOG_ERROR("Unknown MemoryAccessPattern\n");
            assert(false);
        }
    }

    // Check if Phase1 or Phase2 in advance
    bool if_phase1 = false;
    if (contains_limbwise) {
        if (contains_ntt1) {
            if (contains_ntt2) {
                LOG_ERROR(
                    "Unknown SubgraphType: Both NTTPhase1 and 2 is "
                    "contained to one subgraph.\n");
                assert(false);
            }
            if_phase1 = true;
        } else if (contains_ntt2) {
            if_phase1 = false;
        } else {
            LOG_ERROR(
                "Unknown SubgraphType: None of NTTPhase1 and 2 is "
                "contained to one subgraph.\n");
        }
    }

    //
    if (contains_slotwise) {
        if (contains_limbwise) {
            return if_phase1 ? core::SubgraphType::ElemLimb1Slot
                             : core::SubgraphType::ElemLimb2Slot;
        } else {
            return core::SubgraphType::ElemSlot;
        }
    } else if (contains_limbwise) {
        return if_phase1 ? core::SubgraphType::ElemLimb1
                         : core::SubgraphType::ElemLimb2;
    } else if (contains_elemwise) {
        return core::SubgraphType::Elem;
    } else if (only_not_defined) {
        return core::SubgraphType::NoAccess;
    } else {
        LOG_ERROR("Unknown SubgraphType\n");
        assert(false);
    }
}

uint64_t NTTSampleSize(const uint64_t logN) { return 1 << (logN / 2); }

int GetsPolySize(std::vector<std::shared_ptr<polyfhe::core::Node>> &subgraph,
                 std::shared_ptr<Config> config) {
    int spoly_size = 0;

    // Get sPoly Type
    SubgraphType subgraph_type = GetSubgraphType(subgraph);

    // Get max limb
    int max_limb = 0;
    for (auto node : subgraph) {
        for (auto edge : node->get_in_edges()) {
            if (edge->get_level() == polyfhe::core::EdgeLevel::Shared) {
                if (edge->get_limb() > max_limb) {
                    max_limb = edge->get_limb();
                }
            }
        }
        for (auto edge : node->get_out_edges()) {
            if (edge->get_level() == polyfhe::core::EdgeLevel::Shared) {
                if (edge->get_limb() > max_limb) {
                    max_limb = edge->get_limb();
                }
            }
        }
    }

    // Calculate sPoly Size
    switch (subgraph_type) {
    case SubgraphType::Elem:
        spoly_size = 128;
        break;
    case SubgraphType::ElemLimb1:
        spoly_size = NTTSampleSize(config->logN);
        break;
    case SubgraphType::ElemLimb2:
        spoly_size = config->N / NTTSampleSize(config->logN);
        break;
    case SubgraphType::ElemSlot:
        spoly_size = max_limb * 128;
        break;
    case SubgraphType::ElemLimb1Slot:
        spoly_size = NTTSampleSize(config->logN) * max_limb;
        break;
    case SubgraphType::ElemLimb2Slot:
        spoly_size = config->N / NTTSampleSize(config->logN) * max_limb;
        break;
    case SubgraphType::NoAccess:
        spoly_size = 0;
        break;
    default:
        LOG_ERROR("Invalid SubgraphType\n");
        assert(false);
    }
    return spoly_size * sizeof(uint64_t);
}

int GetSubgraphSmemFoorprint(
    std::vector<std::shared_ptr<polyfhe::core::Node>> &subgraph,
    std::shared_ptr<Config> config) {
    LOG_INFO("Calculate Footprint for: ");
    for (auto node : subgraph) {
        std::cout << node->get_op_name() << ", ";
    }
    std::cout << std::endl;

    // Get sPoly Size
    int spoly_size = core::GetsPolySize(subgraph, config);

    // Analyze each outedge can be overwritten or not
    // Only if outedge is one, it can be overwritten

    for (auto node : subgraph) {
        const int n_outedges = node->get_out_edges().size();
        if (n_outedges == 1) {
            auto edge = node->get_out_edges()[0];
            edge->set_can_overwrite(true);
        } else {
            for (int i = 0; i < n_outedges; i++) {
                auto edge = node->get_out_edges()[i];
                edge->set_can_overwrite(false);
            }
        }
    }

    // Get number of sPoly
    int n_spoly = 1;
    for (auto node : subgraph) {
        bool can_overwrite = false;
        bool all_global = true;
        for (auto inedge : node->get_in_edges()) {
            if (inedge->get_level() == polyfhe::core::EdgeLevel::Shared) {
                if (inedge->can_overwrite()) {
                    can_overwrite = true;
                }
            }
        }

        for (auto outedge : node->get_out_edges()) {
            if (outedge->get_level() != polyfhe::core::EdgeLevel::Global) {
                all_global = false;
            }
        }

        if (can_overwrite | all_global) {
            continue;
        } else {
            n_spoly++;
        }
    }

    const int smem_footprint = spoly_size * n_spoly;
    LOG_INFO("-> sPoly %.2f KB * %d = %.2f KB\n", spoly_size / 1000.0, n_spoly,
             smem_footprint / 1000.0);
    return smem_footprint;
}

void ExtractSubgraph(
    std::shared_ptr<polyfhe::core::Node> node,
    std::vector<std::shared_ptr<polyfhe::core::Node>> &subgraph) {
    if (node->get_op_type() == core::OpType::Init ||
        node->get_op_type() == core::OpType::End) {
        return;
    }
    subgraph.push_back(node);
    for (auto edge : node->get_out_edges()) {
        auto found =
            std::find(subgraph.begin(), subgraph.end(), edge->get_dst());
        if (found != subgraph.end()) {
            continue;
        }
        if (edge->get_level() == polyfhe::core::EdgeLevel::Shared) {
            ExtractSubgraph(edge->get_dst(), subgraph);
        }
    }
    for (auto edge : node->get_in_edges()) {
        auto found =
            std::find(subgraph.begin(), subgraph.end(), edge->get_src());
        if (found != subgraph.end()) {
            continue;
        }
        if (edge->get_level() == polyfhe::core::EdgeLevel::Shared) {
            ExtractSubgraph(edge->get_src(), subgraph);
        }
    }
}

} // namespace core
} // namespace polyfhe