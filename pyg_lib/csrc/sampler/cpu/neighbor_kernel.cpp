#include <algorithm>
#include <iterator>

#include <ATen/ATen.h>
#include <omp.h>
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

namespace {

// Multithreading utils
template <typename node_t>
class MTUtils {
 public:
  MTUtils(const int seed_size) : seed_size(seed_size) {
      const int requested_num_threads = 32;
      if (requested_num_threads >= seed_size) {
        seeds_per_thread = 1;
        num = seed_size;
      } else if (seed_size % requested_num_threads != 0) {
        seeds_per_thread = seed_size / requested_num_threads + 1;
        num = seed_size / seeds_per_thread + 1;
      } else {
        seeds_per_thread = seed_size / requested_num_threads;
        num = requested_num_threads;
      }

      // 2. Each thread is assigned a set of nodes for which it looks for
      // neighbors.
      for (auto tid = 0; tid < num; tid++) {
        scope.push_back(tid * seeds_per_thread);
      }
      scope.push_back(
        std::min(num * seeds_per_thread, seed_size));
      
      omp_set_num_threads(num);
  }

  void set_scope(
      std::vector<std::vector<node_t>>& subgraph_sampled_nodes,
      const size_t begin) {
    // This function purpose is to calculate the nodes range for each thread
    // for the next layer based on the number of sampled nodes in the current
    // layer.
    int batch_id = 0;
    int batch_id_end = seeds_per_thread;
    scope[0] = begin;

    for (int t = 1; t < scope.size(); t++) {
      scope[t] = scope[t - 1];
      for (batch_id; batch_id < batch_id_end; batch_id++) {
        scope[t] += subgraph_sampled_nodes[batch_id].size();
      }
      batch_id = batch_id_end;
      batch_id_end =
          std::min(batch_id_end + seeds_per_thread, seed_size);
    }
  }

  const int seed_size;
  int num;
  int seeds_per_thread;

  std::vector<int> scope;
};

// Helper classes for bipartite neighbor sampling //////////////////////////////

// `node_t` is either a scalar or a pair of scalars (example_id, node_id):
template <typename node_t,
          typename scalar_t,
          typename temporal_t,
          bool replace,
          bool save_edges,
          bool save_edge_ids>
class NeighborSampler {
 public:
  NeighborSampler(const scalar_t* rowptr,
                  const scalar_t* col,
                  const std::string temporal_strategy)
      : rowptr_(rowptr), col_(col), temporal_strategy_(temporal_strategy) {
    TORCH_CHECK(temporal_strategy == "uniform" || temporal_strategy == "last",
                "No valid temporal strategy found");
  }

  void allocate_resources(const std::vector<node_t>& global_src_nodes,
                          const std::vector<scalar_t>& seed_times,
                          const c10::optional<at::Tensor>& time,
                          size_t begin,
                          size_t end,
                          int64_t count,
                          const int num_threads,
                          std::vector<int>& threads_scope) {
    if (!save_edges)
      return;
    threads_offsets_.resize(num_threads + 1);

    for (int tid = 1; tid < threads_offsets_.size(); tid++) {
      scalar_t allocation_size = get_allocation_size_(
          global_src_nodes, seed_times, time, threads_scope[tid - 1],
          threads_scope[tid], count);
      threads_offsets_[tid] = threads_offsets_[tid - 1] + allocation_size;
    }

    sampled_id_offset_ = sampled_rows_.size();

    scalar_t size = threads_offsets_.back();
    sampled_rows_.resize(sampled_id_offset_ + size);
    sampled_cols_.resize(sampled_id_offset_ + size);
    if (save_edge_ids) {
      sampled_edge_ids_.resize(sampled_id_offset_ + size);
    }
  }

  void uniform_sample(const node_t global_src_node,
                      const scalar_t local_src_node,
                      const int64_t count,
                      pyg::sampler::Mapper<node_t, scalar_t>& dst_mapper,
                      pyg::random::RandintEngine<scalar_t>& generator,
                      std::vector<node_t>& out_global_dst_nodes,
                      int* node_counter = nullptr) {
    const auto row_start = rowptr_[to_scalar_t(global_src_node)];
    const auto row_end = rowptr_[to_scalar_t(global_src_node) + 1];

    _sample(global_src_node, local_src_node, row_start, row_end, count,
            dst_mapper, generator, out_global_dst_nodes, node_counter);
  }

  void temporal_sample(const node_t global_src_node,
                       const scalar_t local_src_node,
                       const int64_t count,
                       const scalar_t seed_time,
                       const scalar_t* time,
                       pyg::sampler::Mapper<node_t, scalar_t>& dst_mapper,
                       pyg::random::RandintEngine<scalar_t>& generator,
                       std::vector<node_t>& out_global_dst_nodes,
                       int* node_counter = nullptr) {
    const auto row_bounds = get_temporal_neighborhood_bounds(
        global_src_node, count, seed_time, time);

    _sample(global_src_node, local_src_node, row_bounds.first,
            row_bounds.second, count, dst_mapper, generator,
            out_global_dst_nodes, node_counter);
  }

  std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>
  get_sampled_edges(bool csc = false) {
    TORCH_CHECK(save_edges, "No edges have been stored")
    const auto row = pyg::utils::from_vector(sampled_rows_);
    const auto col = pyg::utils::from_vector(sampled_cols_);
    c10::optional<at::Tensor> edge_id = c10::nullopt;
    if (save_edge_ids) {
      edge_id = pyg::utils::from_vector(sampled_edge_ids_);
    }
    if (!csc) {
      return std::make_tuple(row, col, edge_id);
    } else {
      return std::make_tuple(col, row, edge_id);
    }
  }

  void fill_sampled_cols(Mapper<node_t, scalar_t>& mapper,
                         int& node_counter,
                         int& curr,
                         const int sampled_num_thread, size_t sub_num) {
    const int tid = omp_get_thread_num();
    // Keep adding curr values to sampled_cols until encounter a node that
    // has been re-sampled. Put its index into sampled_cols, and so on.
    for (const auto& resampled : mapper.resampled_map) {
      for (node_counter; node_counter < resampled.first; node_counter++) {
        sampled_cols_[sampled_id_offset_ + threads_offsets_[tid] +
                      node_counter] = curr;
        ++curr;
      }
      auto local_node_id = mapper.map(resampled.second);
      sampled_cols_[sampled_id_offset_ + threads_offsets_[tid] + node_counter] =
          local_node_id;
      ++node_counter;
    }
    for (node_counter; node_counter < sampled_num_thread; node_counter++) {
      sampled_cols_[sampled_id_offset_ + threads_offsets_[tid] + node_counter] =
          curr;
      ++curr;
    }

    mapper.resampled_map.clear();
  }

 private:
  std::pair<scalar_t, scalar_t> get_temporal_neighborhood_bounds(
      const node_t global_src_node,
      const int64_t count,
      const scalar_t seed_time,
      const scalar_t* time) {
    auto row_start = rowptr_[to_scalar_t(global_src_node)];
    auto row_end = rowptr_[to_scalar_t(global_src_node) + 1];

    // Find new `row_end` such that all neighbors fulfill temporal constraints:
    auto it = std::upper_bound(
        col_ + row_start, col_ + row_end, seed_time,
        [&](const scalar_t& a, const scalar_t& b) { return a < time[b]; });
    row_end = it - col_;

    if (temporal_strategy_ == "last") {
      row_start = std::max(row_start, (scalar_t)(row_end - count));
    }

    if (row_end - row_start > 1) {
      TORCH_CHECK(time[col_[row_start]] <= time[col_[row_end - 1]],
                  "Found invalid non-sorted temporal neighborhood");
    }
    return std::make_pair(row_start, row_end);
  }

  scalar_t get_allocation_size_(const std::vector<node_t>& global_src_nodes,
                                const std::vector<scalar_t>& seed_times,
                                const c10::optional<at::Tensor>& time,
                                size_t begin,
                                size_t end,
                                int64_t count) {
    scalar_t sum = 0;

    if (!time.has_value()) {
#pragma omp parallel for reduction(+ : sum)
      for (size_t i = begin; i < end; ++i) {
        const auto population = rowptr_[to_scalar_t(global_src_nodes[i]) + 1] -
                                rowptr_[to_scalar_t(global_src_nodes[i])];
        sum += (count < 0 || count > population) ? population : count;
      }
    } else if constexpr (!std::is_scalar<node_t>::value) {  // Temporal:
      const auto time_data = time.value().data_ptr<temporal_t>();
#pragma omp parallel for reduction(+ : sum)
      for (size_t i = begin; i < end; ++i) {
        const auto batch_id = global_src_nodes[i].first;

        const auto row_bounds = get_temporal_neighborhood_bounds(
            global_src_nodes[i], count, seed_times[batch_id], time_data);

        const auto population = row_bounds.second - row_bounds.first;
        sum += (count < 0 || count > population) ? population : count;
      }
    }
    return sum;
  }

  inline scalar_t to_scalar_t(const scalar_t& node) { return node; }
  inline scalar_t to_scalar_t(const std::pair<scalar_t, scalar_t>& node) {
    return std::get<1>(node);
  }

  inline scalar_t to_node_t(const scalar_t& node, const scalar_t& ref) {
    return node;
  }
  inline std::pair<scalar_t, scalar_t> to_node_t(
      const scalar_t& node,
      const std::pair<scalar_t, scalar_t>& ref) {
    return {std::get<0>(ref), node};
  }

  void _sample(const node_t global_src_node,
               const scalar_t local_src_node,
               const scalar_t row_start,
               const scalar_t row_end,
               const int64_t count,
               pyg::sampler::Mapper<node_t, scalar_t>& dst_mapper,
               pyg::random::RandintEngine<scalar_t>& generator,
               std::vector<node_t>& out_global_dst_nodes,
               int* node_counter = nullptr) {
    if (count == 0)
      return;

    const auto population = row_end - row_start;  // liczba sąsiadow

    if (population == 0)
      return;

    // Case 1: Sample the full neighborhood:
    if (count < 0 || (!replace && count >= population)) {
      for (scalar_t edge_id = row_start; edge_id < row_end; ++edge_id) {
        add(edge_id, global_src_node, local_src_node, dst_mapper,
            out_global_dst_nodes, node_counter);
      }
    }

    // Case 2: Sample with replacement:
    else if (replace) {
      for (size_t i = 0; i < count; ++i) {
        const auto edge_id = generator(row_start, row_end);
        add(edge_id, global_src_node, local_src_node, dst_mapper,
            out_global_dst_nodes, node_counter);
      }
    }

    // Case 3: Sample without replacement:
    else {
      auto index_tracker = IndexTracker<scalar_t>(population);
      size_t beg_pos = population - count;
      for (size_t i = beg_pos; i < population; ++i) {
        auto rnd = generator(0, i + 1);
        if (!index_tracker.try_insert(rnd)) {
          rnd = i;
          index_tracker.insert(i);
        }
        const auto edge_id = row_start + rnd;
        add(edge_id, global_src_node, local_src_node, dst_mapper,
            out_global_dst_nodes, node_counter);
      }
    }
  }

  inline void add(const scalar_t edge_id,
                  const node_t global_src_node,
                  const scalar_t local_src_node,
                  pyg::sampler::Mapper<node_t, scalar_t>& dst_mapper,
                  std::vector<node_t>& out_global_dst_nodes,
                  int* node_counter = nullptr) {
    const auto global_dst_node_value = col_[edge_id];
    const auto global_dst_node =
        to_node_t(global_dst_node_value, global_src_node);
    const bool parallel = node_counter != nullptr;
    if (!parallel) {
      const auto res = dst_mapper.insert(global_dst_node);
      if (res.second) {  // not yet sampled.
        out_global_dst_nodes.push_back(global_dst_node);
      }
      if (save_edges) {
        sampled_rows_.push_back(local_src_node);
        sampled_cols_.push_back(res.first);
        if (save_edge_ids) {
          sampled_edge_ids_.push_back(edge_id);
        }
      }
    } else {
      const auto res = dst_mapper.insert(global_dst_node, *node_counter);
      if (res.second) {
        out_global_dst_nodes.push_back(global_dst_node);
      }
      if (save_edges) {
        const int tid = omp_get_thread_num();
        sampled_rows_[sampled_id_offset_ + threads_offsets_[tid] +
                      *node_counter] = local_src_node;

        if (save_edge_ids) {
          sampled_edge_ids_[sampled_id_offset_ + threads_offsets_[tid] +
                            *node_counter] = edge_id;
        }
      }
      ++(*node_counter);
    }
  }

  int64_t sampled_id_offset_ = 0;
  std::vector<scalar_t> threads_offsets_ = {0};

  const scalar_t* rowptr_;
  const scalar_t* col_;
  const std::string temporal_strategy_;
public:
  std::vector<scalar_t> sampled_cols_;
  std::vector<scalar_t> sampled_rows_;
  std::vector<scalar_t> sampled_edge_ids_;
};

// Homogeneous neighbor sampling ///////////////////////////////////////////////
template <typename node_t,
          typename scalar_t,
          typename temporal_t,
          typename NeighborSamplerImpl,
          bool replace,
          bool directed,
          bool disjoint,
          bool return_edge_id>
void sample_seq(NeighborSamplerImpl& sampler,
                std::vector<node_t>& sampled_nodes,
                const at::Tensor& rowptr,
                const at::Tensor& col,
                const at::Tensor& seed,
                const std::vector<int64_t>& num_neighbors,
                const c10::optional<at::Tensor>& time,
                const c10::optional<at::Tensor>& seed_time,
                const bool csc,
                const std::string temporal_strategy) {
  pyg::random::RandintEngine<scalar_t> generator;

  auto mapper = Mapper<node_t, scalar_t>(/*num_nodes=*/rowptr.size(0) - 1);
  std::vector<temporal_t> seed_times;

  const auto seed_data = seed.data_ptr<scalar_t>();
  if constexpr (!disjoint && std::is_scalar<node_t>::value) {
    sampled_nodes = pyg::utils::to_vector<scalar_t>(seed);
    mapper.fill(seed);
  } else {
    for (size_t i = 0; i < seed.numel(); ++i) {
      sampled_nodes.push_back({i, seed_data[i]});
      mapper.insert({i, seed_data[i]});
    }
    if (seed_time.has_value()) {
      const auto seed_time_data = seed_time.value().data_ptr<temporal_t>();
      for (size_t i = 0; i < seed.numel(); ++i) {
        seed_times.push_back(seed_time_data[i]);
      }
    } else if (time.has_value()) {
      const auto time_data = time.value().data_ptr<temporal_t>();
      for (size_t i = 0; i < seed.numel(); ++i) {
        seed_times.push_back(time_data[seed_data[i]]);
      }
    }
  }

  size_t begin = 0, end = seed.size(0);
  for (size_t ell = 0; ell < num_neighbors.size(); ++ell) {
    const auto count = num_neighbors[ell];

    if (!time.has_value()) {
      for (size_t i = begin; i < end; ++i) {
        sampler.uniform_sample(/*global_src_node=*/sampled_nodes[i],
                               /*local_src_node=*/i, count, mapper, generator,
                               /*out_global_dst_nodes=*/sampled_nodes);
      }
    } else if constexpr (!std::is_scalar<node_t>::value) {  // Temporal:
      const auto time_data = time.value().data_ptr<temporal_t>();
      for (size_t i = begin; i < end; ++i) {
        const auto batch_idx = sampled_nodes[i].first;
        sampler.temporal_sample(/*global_src_node=*/sampled_nodes[i],
                                /*local_src_node=*/i, count,
                                seed_times[batch_idx], time_data, mapper,
                                generator,
                                /*out_global_dst_nodes=*/sampled_nodes);
      }
    }
    begin = end, end = sampled_nodes.size();
  }
}

template <typename node_t,
          typename scalar_t,
          typename temporal_t,
          typename NeighborSamplerImpl,
          bool replace,
          bool directed,
          bool disjoint,
          bool return_edge_id>
void sample_parallel(NeighborSamplerImpl& sampler,
                     std::vector<node_t>& sampled_nodes,
                     const at::Tensor& rowptr,
                     const at::Tensor& col,
                     const at::Tensor& seed,
                     const std::vector<int64_t>& num_neighbors,
                     const c10::optional<at::Tensor>& time,
                     const c10::optional<at::Tensor>& seed_time,
                     const bool csc,
                     const std::string temporal_strategy) {
  // In order for the sampling to take place in parallel, it is necessary to
  // ensure that individual threads have their own places in memory, on which
  // they can operate. Therefore, some modifications had to be made to the
  // standard disjont flow:
  // 1. Create separate mappers for each disjoint subgraph.
  // 2. Each thread is assigned a set of nodes for which it looks for neighbors.
  // One thread can work on many mappers, but only one thread can work on
  // one mapper.
  // 3. After sampling the nodes in a given layer, it is required to update the
  // local map values so that they match between all mappers.
  // 4. Finally, fill in the sampled_cols vector based on the updated values in
  // the mappers.

  std::vector<scalar_t> seed_times;
  pyg::random::RandintEngine<scalar_t> generator;

  const auto seed_size = seed.size(0);

  // 1. Create separate mappers for each disjoint subgraph
  std::vector<Mapper<node_t, scalar_t>> mappers(
      seed_size, Mapper<node_t, scalar_t>(/*num_nodes=*/rowptr.size(0) - 1));

  const auto seed_data = seed.data_ptr<scalar_t>();
  if constexpr (!std::is_scalar<node_t>::value) {
    for (size_t i = 0; i < seed.numel(); ++i) {
      sampled_nodes.push_back({i, seed_data[i]});
      mappers[i].curr = i;
      mappers[i].insert({i, seed_data[i]});
    }
  }
  if (seed_time.has_value()) {
    const auto seed_time_data = seed_time.value().data_ptr<scalar_t>();
    for (size_t i = 0; i < seed.numel(); ++i) {
      seed_times.push_back(seed_time_data[i]);
    }
  } else if (time.has_value()) {
    const auto time_data = time.value().data_ptr<scalar_t>();
    for (size_t i = 0; i < seed.numel(); ++i) {
      seed_times.push_back(time_data[seed_data[i]]);
    }
  }

  auto threads = MTUtils<node_t>(static_cast<int>(seed_size));

  size_t begin = 0, end = seed_size;

  for (size_t ell = 0; ell < num_neighbors.size(); ++ell) {
    const auto count = num_neighbors[ell];

    // preparation for going parallel
    sampler.allocate_resources(sampled_nodes, seed_times, time, begin, end,
                               count, threads.num, threads.scope);
    std::vector<std::vector<node_t>> subgraph_sampled_nodes(seed_size);

    std::vector<int> sampled_num_by_prev_subgraphs{0};

#pragma omp parallel num_threads(threads.num)
    {
      const int tid = omp_get_thread_num();
      int node_counter = 0;
      int batch_id = 0;

      if (!time.has_value()) {
        for (auto i = threads.scope[tid]; i < threads.scope[tid + 1]; i++) {
          if constexpr (!std::is_scalar<node_t>::value) {
            batch_id = sampled_nodes[i].first;
          }
          sampler.uniform_sample(
              /*global_src_node=*/sampled_nodes[i],
              /*local_src_node=*/i, count, mappers[batch_id], generator,
              /*out_global_dst_nodes=*/subgraph_sampled_nodes[batch_id],
              &node_counter);
        }
      } else if constexpr (!std::is_scalar<node_t>::value) {  // Temporal:
        const auto time_data = time.value().data_ptr<temporal_t>();
        // Each thread is assigned a set of nodes for which it looks for neighbors.
        for (auto i = threads.scope[tid]; i < threads.scope[tid + 1]; i++) {
          const auto batch_id = sampled_nodes[i].first;

          sampler.temporal_sample(
              /*global_src_node=*/sampled_nodes[i],
              /*local_src_node=*/i, count, seed_times[batch_id], time_data,
              mappers[batch_id], generator,
              /*out_global_dst_nodes=*/subgraph_sampled_nodes[batch_id],
              &node_counter);
        }
      }

// For each subgraph calculate the number of nodes sampled by all previous
// subgraphs.
#pragma omp single
{
  for (int batch_id = 1; batch_id < seed_size; batch_id++) {
    sampled_num_by_prev_subgraphs.push_back(
        sampled_num_by_prev_subgraphs[batch_id - 1] +
        subgraph_sampled_nodes[batch_id - 1].size());
  }
}

// 3. Due to the fact that each subgraph operates on a separate mapper, it is
// required to update the local map values so that they match between all
// mappers.
#pragma omp for
      for (auto batch_id = 0; batch_id < seed_size; batch_id++) {
        mappers[batch_id].update_local_vals(
            sampled_nodes.size(), sampled_num_by_prev_subgraphs[batch_id],
            subgraph_sampled_nodes[batch_id]);
      }

      node_counter = 0;
      int sampled_num_thread = 0;
      int curr = 0;

// 4. Fill sampled cols
#pragma omp for schedule(static, threads.seeds_per_thread)
      for (auto batch_id = 0; batch_id < seed_size; batch_id++) {
        // Initial value of the node index is equal to the sum of all
        // nodes sampled so far (in previous layers) and the nodes sampled by
        // previous subgraphs in the current layer.
        curr = sampled_nodes.size() + sampled_num_by_prev_subgraphs[batch_id];

        // The number of nodes to add to the sampled_cols. It is equal to the
        // number of nodes sampled in a given layer by a given subgraph
        // + the number of re-sampled nodes.
        sampled_num_thread += subgraph_sampled_nodes[batch_id].size() +
                              mappers[batch_id].resampled_map.size();

        sampler.fill_sampled_cols(mappers[batch_id], node_counter, curr,
                                  sampled_num_thread, subgraph_sampled_nodes[batch_id].size());
      }
    }

    for (int i = 0; i < subgraph_sampled_nodes.size(); ++i) {
      std::copy(subgraph_sampled_nodes[i].begin(),
                subgraph_sampled_nodes[i].end(),
                std::back_inserter(sampled_nodes));
    }

    begin = end, end = sampled_nodes.size();

    if (ell < num_neighbors.size() - 1) {
      // No need to calculate new range of nodes for the threads in the last
      // layer.
      threads.set_scope(subgraph_sampled_nodes, begin);
    }
  }
}

// Homogeneous neighbor sampling
/////////////////////////////////////////////////

template <bool replace, bool directed, bool disjoint, bool return_edge_id>
std::tuple<at::Tensor, at::Tensor, at::Tensor, c10::optional<at::Tensor>>
sample(const at::Tensor& rowptr,
       const at::Tensor& col,
       const at::Tensor& seed,
       const std::vector<int64_t>& num_neighbors,
       const c10::optional<at::Tensor>& time,
       const c10::optional<at::Tensor>& seed_time,
       const bool csc,
       const std::string temporal_strategy) {
  TORCH_CHECK(!time.has_value() || disjoint,
              "Temporal sampling needs to create disjoint subgraphs");

  TORCH_CHECK(rowptr.is_contiguous(), "Non-contiguous 'rowptr'");
  TORCH_CHECK(col.is_contiguous(), "Non-contiguous 'col'");
  TORCH_CHECK(seed.is_contiguous(), "Non-contiguous 'seed'");
  if (time.has_value()) {
    TORCH_CHECK(time.value().is_contiguous(), "Non-contiguous 'time'");
  }
  if (seed_time.has_value()) {
    TORCH_CHECK(seed_time.value().is_contiguous(),
                "Non-contiguous 'seed_time'");
  }

  at::Tensor out_row, out_col, out_node_id;
  c10::optional<at::Tensor> out_edge_id = c10::nullopt;

  AT_DISPATCH_INTEGRAL_TYPES(seed.scalar_type(), "sample_kernel", [&] {
    using scalar_t = int64_t;
    typedef std::pair<scalar_t, scalar_t> pair_scalar_t;
    typedef std::conditional_t<!disjoint, scalar_t, pair_scalar_t> node_t;
    typedef int64_t temporal_t;
    typedef NeighborSampler<node_t, scalar_t, temporal_t, replace, directed,
                            return_edge_id>
        NeighborSamplerImpl;

    std::vector<node_t> sampled_nodes;
    auto sampler =
        NeighborSamplerImpl(rowptr.data_ptr<scalar_t>(),
                            col.data_ptr<scalar_t>(), temporal_strategy);

    const bool parallel = true;//disjoint && omp_get_max_threads() > 1 && num_neighbors.size() > 1;
    // std::cout<<"disjoint="<<disjoint<<std::endl;
    // std::cout<<"max threads="<<omp_get_max_threads()<<std::endl;
    // std::cout<<"nn size="<<num_neighbors.size()<<std::endl;
    if (!parallel) {
      sample_seq<node_t, scalar_t, temporal_t, NeighborSamplerImpl, replace,
                 directed, disjoint, return_edge_id>(
          sampler, sampled_nodes, rowptr, col, seed, num_neighbors, time,
          seed_time, csc, temporal_strategy);
    } else {
      sample_parallel<node_t, scalar_t, temporal_t, NeighborSamplerImpl,
                      replace, directed, disjoint, return_edge_id>(
          sampler, sampled_nodes, rowptr, col, seed, num_neighbors, time,
          seed_time, csc, temporal_strategy);
    }

    out_node_id = pyg::utils::from_vector(sampled_nodes);

    TORCH_CHECK(directed, "Undirected subgraphs not yet supported");
    if (directed) {
      std::tie(out_row, out_col, out_edge_id) = sampler.get_sampled_edges(csc);
    } else {
      TORCH_CHECK(!disjoint, "Disjoint subgraphs not yet supported");
    }
  });

  return std::make_tuple(out_row, out_col, out_node_id, out_edge_id);
}

// Heterogeneous neighbor sampling /////////////////////////////////////////////

template <typename node_t, typename scalar_t, typename NeighborSamplerImpl, bool replace, bool directed, bool disjoint, bool return_edge_id>
void sample_seq(
       phmap::parallel_flat_hash_map<edge_type, NeighborSamplerImpl>& sampler_dict,
       phmap::parallel_flat_hash_map<node_type, std::vector<node_t>>& sampled_nodes_dict,
       const std::vector<node_type>& node_types,
       const std::vector<edge_type>& edge_types,
       const c10::Dict<rel_type, at::Tensor>& rowptr_dict,
       const c10::Dict<rel_type, at::Tensor>& col_dict,
       const c10::Dict<node_type, at::Tensor>& seed_dict,
       const c10::Dict<rel_type, std::vector<int64_t>>& num_neighbors_dict,
       const c10::optional<c10::Dict<node_type, at::Tensor>>& time_dict,
       const c10::optional<c10::Dict<node_type, at::Tensor>>& seed_time_dict,
       const bool csc,
       const std::string temporal_strategy) {
    pyg::random::RandintEngine<scalar_t> generator;

    std::vector<std::pair<int, edge_type>> threads_scope_dict; // thread_id, edge_type
    int i = 0;

    phmap::parallel_flat_hash_map<node_type, size_t> num_nodes_dict;

    for (const auto& k : edge_types) {
      const auto num_nodes = rowptr_dict.at(to_rel_type(k)).size(0) - 1; // liczba nodow dla danego edge type
      num_nodes_dict[!csc ? std::get<0>(k) : std::get<2>(k)] = num_nodes; // liczba nodow dla danego typu noda
      const auto dst = !csc ? std::get<2>(k) : std::get<0>(k);
      auto it = std::find_if(threads_scope_dict.begin(), threads_scope_dict.end(), [&dst, &csc](const auto& p){return ((!csc ? std::get<2>(p.second) : std::get<0>(p.second)) == dst);}); // zrobic ten sam myk co dla num nodes
      if (it != threads_scope_dict.end()) {
        threads_scope_dict.push_back({it->first, k});
      } else {
        threads_scope_dict.push_back({i++, k});
        // dst_sampled_nodes_dict[dst];  // Initialize empty vector.
      }
    }

    for (const auto& kv : seed_dict) {
      const at::Tensor& seed = kv.value();
      if (num_nodes_dict.count(kv.key()) == 0 && seed.numel() > 0) {
        num_nodes_dict[kv.key()] = seed.max().data_ptr<scalar_t>()[0] + 1;
      }
    }

    size_t L = 0;  // num_layers.
    phmap::parallel_flat_hash_map<node_type, Mapper<node_t, scalar_t>> mapper_dict;
    phmap::parallel_flat_hash_map<node_type, std::pair<size_t, size_t>> slice_dict;
    std::vector<scalar_t> seed_times;

    for (const auto& k : node_types) {
      const auto N = num_nodes_dict.count(k) > 0 ? num_nodes_dict.at(k) : 0;
      sampled_nodes_dict[k];  // Initialize empty vector.
      mapper_dict.insert({k, Mapper<node_t, scalar_t>(N)});
      slice_dict[k] = {0, 0};
    }

    for (const auto& k : edge_types) {
      L = std::max(L, num_neighbors_dict.at(to_rel_type(k)).size());
      sampler_dict.insert(
          {k, NeighborSamplerImpl(
                  rowptr_dict.at(to_rel_type(k)).data_ptr<scalar_t>(),
                  col_dict.at(to_rel_type(k)).data_ptr<scalar_t>(),
                  temporal_strategy)});
    }

    scalar_t batch_id = 0;
    for (const auto& kv : seed_dict) {
      const at::Tensor& seed = kv.value();
      slice_dict[kv.key()] = {0, seed.size(0)};

      if constexpr (!disjoint) {
        sampled_nodes_dict[kv.key()] = pyg::utils::to_vector<scalar_t>(seed);
        mapper_dict.at(kv.key()).fill(seed);
      } else {
        auto& sampled_nodes = sampled_nodes_dict.at(kv.key());
        auto& mapper = mapper_dict.at(kv.key());
        const auto seed_data = seed.data_ptr<scalar_t>();
        for (size_t i = 0; i < seed.numel(); ++i) {
          sampled_nodes.push_back({batch_id, seed_data[i]});
          mapper.insert({batch_id, seed_data[i]});
          batch_id++;
        }
        if (seed_time_dict.has_value()) {
          const at::Tensor& seed_time = seed_time_dict.value().at(kv.key());
          const auto seed_time_data = seed_time.data_ptr<scalar_t>();
          for (size_t i = 0; i < seed.numel(); ++i) {
            seed_times.push_back(seed_time_data[i]);
          }
        } else if (time_dict.has_value()) {
          const at::Tensor& time = time_dict.value().at(kv.key());
          const auto time_data = time.data_ptr<scalar_t>();
          for (size_t i = 0; i < seed.numel(); ++i) {
            seed_times.push_back(time_data[seed_data[i]]);
          }
        }
      }
    }

    const int num_threads = std::min(i, omp_get_max_threads());

    size_t begin, end;
    for (size_t ell = 0; ell < L; ++ell) {
      phmap::parallel_flat_hash_map<node_type, std::vector<node_t>> dst_sampled_nodes_dict;
      for (const auto& tk : threads_scope_dict) {
        dst_sampled_nodes_dict[!csc ? std::get<2>(tk.second) : std::get<0>(tk.second)]; // initialize empty vector
      }

  omp_set_num_threads(num_threads);    
#pragma omp parallel num_threads(num_threads) default(none) shared(threads_scope_dict, csc, ell, num_neighbors_dict, sampled_nodes_dict, dst_sampled_nodes_dict, mapper_dict, sampler_dict, slice_dict, time_dict, generator, seed_times) private(begin, end, batch_id)
{
      const int tid = omp_get_thread_num();

      for (const auto tk : threads_scope_dict) {
        if (tk.first != tid) {
          continue;
        }
        // dst_sampled_nodes_dict[!csc ? std::get<2>(tk.second) : std::get<0>(tk.second)];
      // for (const auto& k : edge_types) {
        const auto src = !csc ? std::get<0>(tk.second) : std::get<2>(tk.second);
        const auto dst = !csc ? std::get<2>(tk.second) : std::get<0>(tk.second);
        const auto count = num_neighbors_dict.at(to_rel_type(tk.second))[ell];
        auto src_sampled_nodes = sampled_nodes_dict.at(src); // !!!???
        auto& dst_sampled_nodes = dst_sampled_nodes_dict.at(dst);
        auto& dst_mapper = mapper_dict.at(dst);
        auto& sampler = sampler_dict.at(tk.second);
        // std::string printit_tk = "tid="+std::to_string(tid)+", tk.first="+std::to_string(tk.first)+"tk.second="+std::get<2>(tk.second)+", tk.second="+std::get<0>(tk.second)+"\n";
        // std::cout<<printit_tk;
        std::tie(begin, end) = slice_dict.at(src);

        if (!time_dict.has_value() || !time_dict.value().contains(dst)) {
          // std::cout<<"before"<<std::endl;
          for (size_t i = begin; i < end; ++i) {
            sampler.uniform_sample(/*global_src_node=*/src_sampled_nodes[i],
                                   /*local_src_node=*/i, count, dst_mapper,
                                   generator, dst_sampled_nodes);
          }
          // std::cout<<"after"<<std::endl;
        } else if constexpr (!std::is_scalar<node_t>::value) {  // Temporal:
          const at::Tensor& dst_time = time_dict.value().at(dst);
          const auto dst_time_data = dst_time.data_ptr<scalar_t>();
          for (size_t i = begin; i < end; ++i) {
            batch_id = src_sampled_nodes[i].first;
            sampler.temporal_sample(/*global_src_node=*/src_sampled_nodes[i],
                                    /*local_src_node=*/i, count,
                                    seed_times[batch_id], dst_time_data,
                                    dst_mapper, generator,
                                    dst_sampled_nodes);
          }
        }
      }
      // std::string print_dst_sampled;
      // for (auto& dst_sampled : dst_sampled_nodes_dict) {
      //   print_dst_sampled += "{" + dst_sampled.first + ", ";
      //   // for (auto sec : dst_sampled.second) {
      //   //   print_dst_sampled += ", " + std::to_string(sec) + "}\n";
      //   // }
      // }
      // std::cout<<print_dst_sampled;
}
      // for (const auto& k : edge_types) {
      //   std::cout<<"sampled_rows="<<sampler_dict.at(k).sampled_rows_<<std::endl;
      //   std::cout<<"sampled_cols="<<sampler_dict.at(k).sampled_cols_<<std::endl;
      // }
      for (auto& dst_sampled : dst_sampled_nodes_dict) {
        std::copy(dst_sampled.second.begin(),
                dst_sampled.second.end(),
                std::back_inserter(sampled_nodes_dict[dst_sampled.first]));
      }

    // for (auto& sampled : sampled_nodes_dict) {
    //   std::cout<<"{"<<sampled.first<<": ";
    //   for (auto &v : sampled.second) {
    //     std::cout<<v<<", ";
    //   }
    //   std::cout<<"}"<<std::endl;
    // }

      for (const auto& k : node_types) {
        // std::cout<<"before: k="<<k<<"{"<<slice_dict[k].first<<", "<<slice_dict[k].second<<"}"<<std::endl;
        slice_dict[k] = {slice_dict.at(k).second,
                         sampled_nodes_dict.at(k).size()};
        // std::cout<<"after: k="<<k<<"{"<<slice_dict[k].first<<", "<<slice_dict[k].second<<"}"<<std::endl;
      }
    }
}
// Parallel heterogeneous neighbor sampling ////////////////////////////////////

template <typename node_t, typename scalar_t, typename NeighborSamplerImpl, bool replace, bool directed, bool disjoint, bool return_edge_id>
void sample_parallel(
       phmap::parallel_flat_hash_map<edge_type, NeighborSamplerImpl>& sampler_dict,
       phmap::parallel_flat_hash_map<node_type, std::vector<node_t>>& sampled_nodes_dict,
       const std::vector<node_type>& node_types,
       const std::vector<edge_type>& edge_types,
       const c10::Dict<rel_type, at::Tensor>& rowptr_dict,
       const c10::Dict<rel_type, at::Tensor>& col_dict,
       const c10::Dict<node_type, at::Tensor>& seed_dict,
       const c10::Dict<rel_type, std::vector<int64_t>>& num_neighbors_dict,
       const c10::optional<c10::Dict<node_type, at::Tensor>>& time_dict,
       const c10::optional<c10::Dict<node_type, at::Tensor>>& seed_time_dict,
       const bool csc,
       const std::string temporal_strategy) {
    pyg::random::RandintEngine<scalar_t> generator;

    phmap::parallel_flat_hash_map<node_type, size_t> num_nodes_dict;
    for (const auto& k : edge_types) {
      const auto num_nodes = rowptr_dict.at(to_rel_type(k)).size(0) - 1;
      num_nodes_dict[!csc ? std::get<0>(k) : std::get<2>(k)] = num_nodes;
    }
    for (const auto& kv : seed_dict) {
      const at::Tensor& seed = kv.value();
      if (num_nodes_dict.count(kv.key()) == 0 && seed.numel() > 0) {
        num_nodes_dict[kv.key()] = seed.max().data_ptr<scalar_t>()[0] + 1;
      }
    }

    size_t L = 0;  // num_layers.
    phmap::parallel_flat_hash_map<node_type, Mapper<node_t, scalar_t>> mapper_dict;
    phmap::parallel_flat_hash_map<node_type, std::pair<size_t, size_t>> slice_dict;
    std::vector<scalar_t> seed_times;

    for (const auto& k : node_types) {
      const auto N = num_nodes_dict.count(k) > 0 ? num_nodes_dict.at(k) : 0;
      sampled_nodes_dict[k];  // Initialize empty vector.
      mapper_dict.insert({k, Mapper<node_t, scalar_t>(N)});
      slice_dict[k] = {0, 0};
    }

    for (const auto& k : edge_types) {
      L = std::max(L, num_neighbors_dict.at(to_rel_type(k)).size());
      sampler_dict.insert(
          {k, NeighborSamplerImpl(
                  rowptr_dict.at(to_rel_type(k)).data_ptr<scalar_t>(),
                  col_dict.at(to_rel_type(k)).data_ptr<scalar_t>(),
                  temporal_strategy)});
    }

    scalar_t batch_id = 0;
    for (const auto& kv : seed_dict) {
      const at::Tensor& seed = kv.value();
      slice_dict[kv.key()] = {0, seed.size(0)};

      if constexpr (!disjoint) {
        sampled_nodes_dict[kv.key()] = pyg::utils::to_vector<scalar_t>(seed);
        mapper_dict.at(kv.key()).fill(seed);
      } else {
        auto& sampled_nodes = sampled_nodes_dict.at(kv.key());
        auto& mapper = mapper_dict.at(kv.key());
        const auto seed_data = seed.data_ptr<scalar_t>();
        for (size_t i = 0; i < seed.numel(); ++i) {
          sampled_nodes.push_back({batch_id, seed_data[i]});
          mapper.insert({batch_id, seed_data[i]});
          batch_id++;
        }
        if (seed_time_dict.has_value()) {
          const at::Tensor& seed_time = seed_time_dict.value().at(kv.key());
          const auto seed_time_data = seed_time.data_ptr<scalar_t>();
          for (size_t i = 0; i < seed.numel(); ++i) {
            seed_times.push_back(seed_time_data[i]);
          }
        } else if (time_dict.has_value()) {
          const at::Tensor& time = time_dict.value().at(kv.key());
          const auto time_data = time.data_ptr<scalar_t>();
          for (size_t i = 0; i < seed.numel(); ++i) {
            seed_times.push_back(time_data[seed_data[i]]);
          }
        }
      }
    }

    size_t begin, end;
    for (size_t ell = 0; ell < L; ++ell) {
      for (const auto& k : edge_types) {
        const auto& src = !csc ? std::get<0>(k) : std::get<2>(k);
        const auto& dst = !csc ? std::get<2>(k) : std::get<0>(k);
        const auto count = num_neighbors_dict.at(to_rel_type(k))[ell];
        auto& src_sampled_nodes = sampled_nodes_dict.at(src);
        auto& dst_sampled_nodes = sampled_nodes_dict.at(dst);
        auto& dst_mapper = mapper_dict.at(dst);
        auto& sampler = sampler_dict.at(k);
        std::tie(begin, end) = slice_dict.at(src);

        if (!time_dict.has_value() || !time_dict.value().contains(dst)) {
          for (size_t i = begin; i < end; ++i) {
            sampler.uniform_sample(/*global_src_node=*/src_sampled_nodes[i],
                                   /*local_src_node=*/i, count, dst_mapper,
                                   generator, dst_sampled_nodes);
          }
        } else if constexpr (!std::is_scalar<node_t>::value) {  // Temporal:
          const at::Tensor& dst_time = time_dict.value().at(dst);
          const auto dst_time_data = dst_time.data_ptr<scalar_t>();
          for (size_t i = begin; i < end; ++i) {
            batch_id = src_sampled_nodes[i].first;
            sampler.temporal_sample(/*global_src_node=*/src_sampled_nodes[i],
                                    /*local_src_node=*/i, count,
                                    seed_times[batch_id], dst_time_data,
                                    dst_mapper, generator,
                                    dst_sampled_nodes);
          }
        }
      }
      for (const auto& k : node_types) {
        slice_dict[k] = {slice_dict.at(k).second,
                         sampled_nodes_dict.at(k).size()};
      }
    }
}


// Heterogeneous neighbor sampling /////////////////////////////////////////////

template <bool replace, bool directed, bool disjoint, bool return_edge_id>
std::tuple<c10::Dict<rel_type, at::Tensor>,
           c10::Dict<rel_type, at::Tensor>,
           c10::Dict<node_type, at::Tensor>,
           c10::optional<c10::Dict<rel_type, at::Tensor>>>
sample(const std::vector<node_type>& node_types,
       const std::vector<edge_type>& edge_types,
       const c10::Dict<rel_type, at::Tensor>& rowptr_dict,
       const c10::Dict<rel_type, at::Tensor>& col_dict,
       const c10::Dict<node_type, at::Tensor>& seed_dict,
       const c10::Dict<rel_type, std::vector<int64_t>>& num_neighbors_dict,
       const c10::optional<c10::Dict<node_type, at::Tensor>>& time_dict,
       const c10::optional<c10::Dict<node_type, at::Tensor>>& seed_time_dict,
       const bool csc,
       const std::string temporal_strategy) {
  TORCH_CHECK(!time_dict.has_value() || disjoint,
              "Temporal sampling needs to create disjoint subgraphs");

  for (const auto& kv : rowptr_dict) {
    const at::Tensor& rowptr = kv.value();
    TORCH_CHECK(rowptr.is_contiguous(), "Non-contiguous 'rowptr'");
  }
  for (const auto& kv : col_dict) {
    const at::Tensor& col = kv.value();
    TORCH_CHECK(col.is_contiguous(), "Non-contiguous 'col'");
  }
  for (const auto& kv : seed_dict) {
    const at::Tensor& seed = kv.value();
    TORCH_CHECK(seed.is_contiguous(), "Non-contiguous 'seed'");
  }
  if (time_dict.has_value()) {
    for (const auto& kv : time_dict.value()) {
      const at::Tensor& time = kv.value();
      TORCH_CHECK(time.is_contiguous(), "Non-contiguous 'time'");
    }
  }
  if (seed_time_dict.has_value()) {
    for (const auto& kv : seed_time_dict.value()) {
      const at::Tensor& seed_time = kv.value();
      TORCH_CHECK(seed_time.is_contiguous(), "Non-contiguous 'seed_time'");
    }
  }

  c10::Dict<rel_type, at::Tensor> out_row_dict, out_col_dict;
  c10::Dict<node_type, at::Tensor> out_node_id_dict;
  c10::optional<c10::Dict<node_type, at::Tensor>> out_edge_id_dict;
  if (return_edge_id) {
    out_edge_id_dict = c10::Dict<rel_type, at::Tensor>();
  } else {
    out_edge_id_dict = c10::nullopt;
  }

  const auto scalar_type = seed_dict.begin()->value().scalar_type();
  AT_DISPATCH_INTEGRAL_TYPES(scalar_type, "hetero_sample_kernel", [&] {
    typedef std::pair<scalar_t, scalar_t> pair_scalar_t;
    typedef std::conditional_t<!disjoint, scalar_t, pair_scalar_t> node_t;
    typedef int64_t temporal_t;
    typedef NeighborSampler<node_t, scalar_t, temporal_t, replace, directed,
                            return_edge_id>
        NeighborSamplerImpl;

    phmap::parallel_flat_hash_map<edge_type, NeighborSamplerImpl> sampler_dict;
    phmap::parallel_flat_hash_map<node_type, std::vector<node_t>> sampled_nodes_dict;

    const bool parallel = false;
        //disjoint && omp_get_max_threads() > 1 && num_neighbors.size() > 1;
    if (!parallel) {
      sample_seq<node_t, scalar_t, NeighborSamplerImpl, replace,
                 directed, disjoint, return_edge_id>(
          sampler_dict, sampled_nodes_dict, node_types, edge_types, 
          rowptr_dict, col_dict, seed_dict,
          num_neighbors_dict, time_dict,
          seed_time_dict, csc, temporal_strategy);
    } else {
      // sample_parallel<node_t, scalar_t, temporal_t, NeighborSamplerImpl,
      //                 replace, directed, disjoint, return_edge_id>(
      //     sampler, sampled_nodes, rowptr, col, seed, num_neighbors, time,
      //     seed_time, csc, temporal_strategy);
    }

    for (const auto& k : node_types) {
      out_node_id_dict.insert(
          k, pyg::utils::from_vector(sampled_nodes_dict.at(k)));
    }

    TORCH_CHECK(directed, "Undirected heterogeneous graphs not yet supported");
    if (directed) {
      for (const auto& k : edge_types) {
        const auto edges = sampler_dict.at(k).get_sampled_edges(csc);
        out_row_dict.insert(to_rel_type(k), std::get<0>(edges));
        out_col_dict.insert(to_rel_type(k), std::get<1>(edges));
        if (return_edge_id) {
          out_edge_id_dict.value().insert(to_rel_type(k),
                                          std::get<2>(edges).value());
        }
      }
    }
  });

  return std::make_tuple(out_row_dict, out_col_dict, out_node_id_dict,
                         out_edge_id_dict);
}


// Dispatcher //////////////////////////////////////////////////////////////////

#define DISPATCH_SAMPLE(replace, directed, disjount, return_edge_id, ...) \
  if (replace && directed && disjoint && return_edge_id)                  \
    return sample<true, true, true, true>(__VA_ARGS__);                   \
  if (replace && directed && disjoint && !return_edge_id)                 \
    return sample<true, true, true, false>(__VA_ARGS__);                  \
  if (replace && directed && !disjoint && return_edge_id)                 \
    return sample<true, true, false, true>(__VA_ARGS__);                  \
  if (replace && directed && !disjoint && !return_edge_id)                \
    return sample<true, true, false, false>(__VA_ARGS__);                 \
  if (replace && !directed && disjoint && return_edge_id)                 \
    return sample<true, false, true, true>(__VA_ARGS__);                  \
  if (replace && !directed && disjoint && !return_edge_id)                \
    return sample<true, false, true, false>(__VA_ARGS__);                 \
  if (replace && !directed && !disjoint && return_edge_id)                \
    return sample<true, false, false, true>(__VA_ARGS__);                 \
  if (replace && !directed && !disjoint && !return_edge_id)               \
    return sample<true, false, false, false>(__VA_ARGS__);                \
  if (!replace && directed && disjoint && return_edge_id)                 \
    return sample<false, true, true, true>(__VA_ARGS__);                  \
  if (!replace && directed && disjoint && !return_edge_id)                \
    return sample<false, true, true, false>(__VA_ARGS__);                 \
  if (!replace && directed && !disjoint && return_edge_id)                \
    return sample<false, true, false, true>(__VA_ARGS__);                 \
  if (!replace && directed && !disjoint && !return_edge_id)               \
    return sample<false, true, false, false>(__VA_ARGS__);                \
  if (!replace && !directed && disjoint && return_edge_id)                \
    return sample<false, false, true, true>(__VA_ARGS__);                 \
  if (!replace && !directed && disjoint && !return_edge_id)               \
    return sample<false, false, true, false>(__VA_ARGS__);                \
  if (!replace && !directed && !disjoint && return_edge_id)               \
    return sample<false, false, false, true>(__VA_ARGS__);                \
  if (!replace && !directed && !disjoint && !return_edge_id)              \
    return sample<false, false, false, false>(__VA_ARGS__);

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, c10::optional<at::Tensor>>
neighbor_sample_kernel(const at::Tensor& rowptr,
                       const at::Tensor& col,
                       const at::Tensor& seed,
                       const std::vector<int64_t>& num_neighbors,
                       const c10::optional<at::Tensor>& time,
                       const c10::optional<at::Tensor>& seed_time,
                       bool csc,
                       bool replace,
                       bool directed,
                       bool disjoint,
                       std::string temporal_strategy,
                       bool return_edge_id) {
  DISPATCH_SAMPLE(replace, directed, disjoint, return_edge_id, rowptr, col,
                  seed, num_neighbors, time, seed_time, csc, temporal_strategy);
}

std::tuple<c10::Dict<rel_type, at::Tensor>,
           c10::Dict<rel_type, at::Tensor>,
           c10::Dict<node_type, at::Tensor>,
           c10::optional<c10::Dict<rel_type, at::Tensor>>>
hetero_neighbor_sample_kernel(
    const std::vector<node_type>& node_types,
    const std::vector<edge_type>& edge_types,
    const c10::Dict<rel_type, at::Tensor>& rowptr_dict,
    const c10::Dict<rel_type, at::Tensor>& col_dict,
    const c10::Dict<node_type, at::Tensor>& seed_dict,
    const c10::Dict<rel_type, std::vector<int64_t>>& num_neighbors_dict,
    const c10::optional<c10::Dict<node_type, at::Tensor>>& time_dict,
    const c10::optional<c10::Dict<node_type, at::Tensor>>& seed_time_dict,
    bool csc,
    bool replace,
    bool directed,
    bool disjoint,
    std::string temporal_strategy,
    bool return_edge_id) {
  DISPATCH_SAMPLE(replace, directed, disjoint, return_edge_id, node_types,
                  edge_types, rowptr_dict, col_dict, seed_dict,
                  num_neighbors_dict, time_dict, seed_time_dict, csc,
                  temporal_strategy);
}

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::neighbor_sample"),
         TORCH_FN(neighbor_sample_kernel));
}

 // We use `BackendSelect` as a fallback to the dispatcher logic as automatic
 // dispatching of dictionaries is not yet supported by PyTorch.
 // See: pytorch/aten/src/ATen/templates/RegisterBackendSelect.cpp.
 TORCH_LIBRARY_IMPL(pyg, BackendSelect, m) {
   m.impl(TORCH_SELECTIVE_NAME("pyg::hetero_neighbor_sample"),
          TORCH_FN(hetero_neighbor_sample_kernel));
 }
}  // namespace sampler
}  // namespace pyg
