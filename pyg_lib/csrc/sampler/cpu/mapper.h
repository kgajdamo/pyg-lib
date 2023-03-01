#pragma once

#include <ATen/ATen.h>
#include <omp.h>

#include "parallel_hashmap/btree.h"
#include "parallel_hashmap/phmap.h"

#include <algorithm>
#include <iterator>

#include <torch/library.h>

#include "parallel_hashmap/phmap.h"

#include "pyg_lib/csrc/random/cpu/rand_engine.h"
#include "pyg_lib/csrc/sampler/cpu/index_tracker.h"
#include "pyg_lib/csrc/sampler/cpu/mapper.h"
#include "pyg_lib/csrc/sampler/cpu/neighbor_kernel.h"
#include "pyg_lib/csrc/sampler/subgraph.h"
#include "pyg_lib/csrc/utils/cpu/convert.h"
#include "pyg_lib/csrc/utils/types.h"

namespace pyg {
namespace sampler {

// TODO Implement `Mapper` as an interface/abstract class to allow for other
// implementations as well.
template <typename node_t, typename scalar_t>
class Mapper {
 public:
  Mapper(const size_t num_nodes, const size_t num_entries = -1)
      : num_nodes(num_nodes), num_entries(num_entries) {
    // We use some simple heuristic to determine whether we can use a vector
    // to perform the mapping instead of relying on the more memory-friendly,
    // but slower hash map implementation. As a general rule of thumb, we are
    // safe to use vectors in case the number of nodes are small, or it is
    // expected that we sample a large amount of nodes.
    use_vec = (num_nodes < 1000000) || (num_entries > num_nodes / 10);

    if (num_nodes <= 0) {  // == `num_nodes` is undefined:
      use_vec = false;
    }

    // We can only utilize vector mappings in case entries are scalar:
    if (!std::is_scalar<node_t>::value) {
      use_vec = false;
    }

    if (use_vec) {
      to_local_vec.resize(num_nodes, -1);
    }
  }

  // For parallel disjoint
  std::pair<scalar_t, bool> insert(const node_t& node, int thread_node_counter) {
    auto out = to_local_map.insert({node, curr});
    auto res = std::pair<scalar_t, bool>(out.first->second, out.second);
    if (res.second) {
      ++curr;
    } else {
      // Save the nodes that have been resampled to the map, where the key is
      // the node number sampled by the given thread. This will be needed to
      // fill the sampled_cols vector.
      resampled_map.insert({thread_node_counter, node});
    }
    return res;
  }

  std::pair<scalar_t, bool> insert(const node_t& node) {
    std::pair<scalar_t, bool> res;
    if (use_vec) {
      if constexpr (std::is_scalar<node_t>::value) {
        auto old = to_local_vec[node];
        res = std::pair<scalar_t, bool>(old == -1 ? curr : old, old == -1);
        if (res.second)
          to_local_vec[node] = curr;
      }
    } else {
      auto out = to_local_map.insert({node, curr});
      res = std::pair<scalar_t, bool>(out.first->second, out.second);
    }
    if (res.second) {
      ++curr;
    }
    return res;
  }

  void fill(const node_t* nodes, const size_t size) {
    for (size_t i = 0; i < size; ++i) {
      insert(nodes[i]);
    }
  }

  void fill(const at::Tensor& nodes) {
    fill(nodes.data_ptr<node_t>(), nodes.numel());
  }

  bool exists(const node_t& node) {
    if (use_vec) {
      return to_local_vec[node] >= 0;
    } else {
      return to_local_map.count(node) > 0;
    }
  }

  scalar_t map(const node_t& node) {
    if (use_vec) {
      if constexpr (std::is_scalar<node_t>::value) {
        return to_local_vec[node];
      }
    } else {
      const auto search = to_local_map.find(node);
      return search != to_local_map.end() ? search->second : -1;
    }
  }

  void update_local_vals(size_t sampled_nodes_size,
                         int sampled_num_by_prev_subgraphs,
                         std::vector<node_t>& subgraph_sampled_nodes) {
    // Iterate over sampled nodes to update their local values
    for (const auto& sampled_node : subgraph_sampled_nodes) {
      const auto search = to_local_map.find(sampled_node);
      if (search != to_local_map.end()) {
        search->second += sampled_nodes_size - curr +
                          subgraph_sampled_nodes.size() +
                          sampled_num_by_prev_subgraphs;
      }
    }
  }

 private:
  const size_t num_nodes, num_entries;

  bool use_vec;
  std::vector<scalar_t> to_local_vec;
  phmap::flat_hash_map<node_t, scalar_t> to_local_map;

 public:
  scalar_t curr = 0;
  phmap::btree_map<int, node_t> resampled_map;
};

}  // namespace sampler
}  // namespace pyg
