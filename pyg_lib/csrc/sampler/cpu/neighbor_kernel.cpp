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
                          std::vector<int>& threads_ranges) {
    if (!save_edges)
      return;
    // std::cout<<"sampled_rows="<<sampled_rows_<<std::endl;
    // std::cout<<"sampled_cols="<<sampled_cols_<<std::endl;
    threads_offsets_.resize(num_threads + 1);

    for (int tid = 1; tid < threads_offsets_.size(); tid++) {
      scalar_t allocation_size = get_allocation_size_(
          global_src_nodes, seed_times, time, threads_ranges[tid - 1],
          threads_ranges[tid], count);
      threads_offsets_[tid] = threads_offsets_[tid - 1] + allocation_size;
    }

    sampled_id_offset_ = sampled_rows_.size();

    scalar_t size = threads_offsets_.back();
    sampled_rows_.resize(sampled_id_offset_ + size);
    sampled_cols_.resize(sampled_id_offset_ + size);
    if (save_edge_ids) {
      sampled_edge_ids_.resize(sampled_id_offset_ + size);
    }

    // std::cout<<"threads_offsets_="<<threads_offsets_<<std::endl;
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
    auto row_start = rowptr_[to_scalar_t(global_src_node)];
    auto row_end = rowptr_[to_scalar_t(global_src_node) + 1];

    // Find new `row_end` such that all neighbors fulfill temporal constraints:
    auto it = std::lower_bound(
        col_ + row_start, col_ + row_end, seed_time,
        [&](const scalar_t& a, const scalar_t& b) { return time[a] < b; });
    row_end = it - col_;

    if (temporal_strategy_ == "last") {
      row_start = std::max(row_start, (scalar_t)(row_end - count));
    }

    if (row_end - row_start > 1) {
      TORCH_CHECK(time[col_[row_start]] <= time[col_[row_end - 1]],
                  "Found invalid non-sorted temporal neighborhood");
    }

    _sample(global_src_node, local_src_node, row_start, row_end, count,
            dst_mapper, generator, out_global_dst_nodes, node_counter);
  }

  std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>
  get_sampled_edges(bool csc = false) {
    TORCH_CHECK(save_edges, "No edges have been stored")
    // std::cout<<"sampled_rows_="<<sampled_rows_<<std::endl;
    std::cout<<"sampled_cols_="<<sampled_cols_<<std::endl;
    const auto row = pyg::utils::from_vector(sampled_rows_);
    const auto col = pyg::utils::from_vector(sampled_cols_);
    c10::optional<at::Tensor> edge_id = c10::nullopt;
    if (save_edge_ids) {
      // std::cout<<"sampled_edge_ids_="<<sampled_edge_ids_<<std::endl;
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
                         const int sampled_num_thread) {
    const int tid = omp_get_thread_num();
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
        auto row_start = rowptr_[to_scalar_t(global_src_nodes[i])];
        auto row_end = rowptr_[to_scalar_t(global_src_nodes[i]) + 1];
        // std::cout<<"first
        // row_start="<<row_start<<"row_end="<<row_end<<std::endl;
        // std::cout<<"seed_times[batch_id]="<<seed_times[batch_id]<<std::endl;

        // Find new `row_end` such that all neighbors fulfill temporal
        // constraints:
        auto it = std::lower_bound(col_ + row_start, col_ + row_end,
                                   seed_times[batch_id],
                                   [&](const scalar_t& a, const scalar_t& b) {
                                     return time_data[a] < b;
                                   });
        row_end = it - col_;

        if (temporal_strategy_ == "last") {
          row_start = std::max(row_start, (scalar_t)(row_end - count));
        }

        if (row_end - row_start > 1) {
          TORCH_CHECK(
              time_data[col_[row_start]] <= time_data[col_[row_end - 1]],
              "Found invalid non-sorted temporal neighborhood");
        }

        // std::string printit = "new row_start="+std::to_string(row_start)+"
        // new row end="+std::to_string(row_end)+"\n"; std::cout<<printit;

        const auto population = row_end - row_start;
        sum += (count < 0 || count > population) ? population : count;
        // std::cout<<"sum="<<sum<<std::endl;
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

    // std::string printit = "tid="+std::to_string(tid) + "
    // node_counter="+std::to_string(node_counter)+"\n"; std::cout<<printit;
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
  // typedef std::pair<scalar_t, scalar_t> pair_scalar_t;
  // typedef std::conditional_t<!disjoint, scalar_t, pair_scalar_t> node_t;

  // std::vector<node_t> sampled_nodes;
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

  // return sampled_nodes;
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
  // typedef std::pair<scalar_t, scalar_t> pair_scalar_t;
  // typedef std::conditional_t<!disjoint, scalar_t, pair_scalar_t> node_t;

  // std::vector<node_t> sampled_nodes;

  std::vector<scalar_t> seed_times;
  pyg::random::RandintEngine<scalar_t> generator;
  std::vector<Mapper<node_t, scalar_t>> mappers(
      seed.size(0), Mapper<node_t, scalar_t>(/*num_nodes=*/rowptr.size(0) - 1));

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

  // NAPISAĆ TEST SPRAWDZAJĄCY POPRAWNOŚĆ WYNIKU, PORÓWNUJĄCY WYNIKI PARALLEL I SEQ
  const int requested_num_threads = omp_get_max_threads();
  // std::cout<<"requested_num_threads="<<requested_num_threads<<std::endl;
  int num_threads = requested_num_threads >= seed.size(0) ? seed.size(0) : requested_num_threads;
  std::cout<<"num_threads="<<num_threads<<std::endl;

  const int seeds_per_thread = seed.size(0) % num_threads == 0
                                   ? seed.size(0) / num_threads
                                   : seed.size(0) / num_threads + 1;
  std::cout<<"seeds_per_thread="<<seeds_per_thread<<std::endl;

  std::vector<int> threads_ranges;
  for (auto tid = 0; tid < num_threads; tid++) {
    threads_ranges.push_back(tid * seeds_per_thread);
  }
  threads_ranges.push_back(
      std::min(num_threads * seeds_per_thread, static_cast<int>(seed.size(0))));
  // std::cout<<"threads_ranges="<<threads_ranges<<std::endl;

  size_t begin = 0, end = seed.size(0);

  for (size_t ell = 0; ell < num_neighbors.size(); ++ell) {
    const auto count = num_neighbors[ell];

    // preparation for going parallel
    sampler.allocate_resources(sampled_nodes, seed_times, time, begin, end,
                               count, num_threads, threads_ranges);
    std::vector<std::vector<node_t>> subgraph_sampled_nodes(mappers.size());

    omp_set_num_threads(num_threads);
#pragma omp parallel num_threads(num_threads)
    {
      const int tid = omp_get_thread_num();
      int node_counter = 0;
      int batch_id = 0;

      if (!time.has_value()) {
        for (auto i = threads_ranges[tid]; i < threads_ranges[tid + 1]; i++) {
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
        for (auto i = threads_ranges[tid]; i < threads_ranges[tid + 1]; i++) {
          const auto batch_id = sampled_nodes[i].first;

          sampler.temporal_sample(
              /*global_src_node=*/sampled_nodes[i],
              /*local_src_node=*/i, count, seed_times[batch_id], time_data,
              mappers[batch_id], generator,
              /*out_global_dst_nodes=*/subgraph_sampled_nodes[batch_id],
              &node_counter);
        }
      }
      // }
      for (int m=0; m<mappers.size(); m++) {
        std::cout<<"resampled"<<std::endl;
        std::cout<<"batch_id="<<m<<std::endl;
            for (auto &x: mappers[m].resampled_map) {
              std::cout<<x.first<<" "<<x.second;
            }
            std::cout<<std::endl;
          }

      // for (int m=0; m<mappers.size(); m++) {
      //   std::cout<<"to_local_map"<<std::endl;
      //   std::cout<<"batch_id="<<m<<std::endl;
      //       for (auto &x: mappers[m].to_local_map) {
      //         std::cout<<x.first<<" "<<x.second;
      //       }
      //       std::cout<<std::endl;
      //     }

      // for (int m=0; m<subgraph_sampled_nodes.size(); m++) {
      //   std::cout<<"subgraph_sampled_nodes"<<std::endl;
      //   std::cout<<"batch_id="<<m<<std::endl;
      //   std::cout<<subgraph_sampled_nodes[m]<<std::endl;
      // }

      std::vector<int> sampled_num_by_prev_subgraphs{0};
#pragma omp single
      {
        for (int batch_id = 1; batch_id < mappers.size(); batch_id++) {
          sampled_num_by_prev_subgraphs.push_back(
              sampled_num_by_prev_subgraphs[batch_id - 1] +
              subgraph_sampled_nodes[batch_id - 1].size());
        }
      }

// update local_map values
// #pragma omp parallel num_threads(num_threads)
//     {
#pragma omp for
      for (auto batch_id = 0; batch_id < mappers.size(); batch_id++) {
        mappers[batch_id].update_local_val(
            sampled_nodes.size(), sampled_num_by_prev_subgraphs[batch_id],
            subgraph_sampled_nodes[batch_id]);
      }

      node_counter = 0;
      int sampled_num_thread = 0;
      int curr = 0;

#pragma omp for schedule(static, seeds_per_thread)
      for (auto batch_id = 0; batch_id < mappers.size(); batch_id++) {
        curr = sampled_num_by_prev_subgraphs[batch_id] +
               threads_ranges[num_threads];
        sampled_num_thread += subgraph_sampled_nodes[batch_id].size() +
                              mappers[batch_id].resampled_map.size();

        // std::string printit = "tid="+std::to_string(tid)+",
        // sampled_num_thread=" + std::to_string(sampled_num_thread)+"\n";
        // std::cout<<printit;
        sampler.fill_sampled_cols(mappers[batch_id], node_counter, curr,
                                  sampled_num_thread);
      }
    }
          for (int m=0; m<mappers.size(); m++) {
        std::cout<<"to_local_map after"<<std::endl;
        std::cout<<"batch_id="<<m<<std::endl;
            for (auto &x: mappers[m].to_local_map) {
              std::cout<<x.first<<" "<<x.second<<" ";
            }
            std::cout<<std::endl;
          }
     std::cout<<"sampled_cols="<<sampler.sampled_cols_<<std::endl;
     std::cout<<"sampled_rows="<<sampler.sampled_rows_<<std::endl;
    for (int i = 0; i < subgraph_sampled_nodes.size(); ++i) {
      std::copy(subgraph_sampled_nodes[i].begin(),
                subgraph_sampled_nodes[i].end(),
                std::back_inserter(sampled_nodes));
    }

    begin = end, end = sampled_nodes.size();

    // std::cout<<"threads_ranges_before="<<threads_ranges<<std::endl;
    int batch_id = 0;
    int batch_id_end = seeds_per_thread;
    threads_ranges[0] = begin;
    for (int t = 1; t < threads_ranges.size(); t++) {
      threads_ranges[t] = threads_ranges[t - 1];
      for (batch_id; batch_id < batch_id_end; batch_id++) {
        threads_ranges[t] += subgraph_sampled_nodes[batch_id].size();
      }
      batch_id = batch_id_end;
      batch_id_end = std::min(batch_id_end + seeds_per_thread,
                              static_cast<int>(seed.size(0)));
    }

    // std::cout<<"threads_ranges_after="<<threads_ranges<<std::endl;
  }

  // return sampled_nodes;
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

    const bool parallel =
        disjoint && omp_get_max_threads() > 1 && num_neighbors.size() > 1;
    if (!parallel) {
      sample_seq<node_t, scalar_t, temporal_t, NeighborSamplerImpl, replace,
                 directed, disjoint, return_edge_id>(
          sampler, sampled_nodes, rowptr, col, seed, num_neighbors, time,
          seed_time, csc, temporal_strategy);
    } else {
      std::cout<<"parallel"<<std::endl;
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

    pyg::random::RandintEngine<scalar_t> generator;

    phmap::flat_hash_map<node_type, size_t> num_nodes_dict;
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
    phmap::flat_hash_map<node_type, std::vector<node_t>> sampled_nodes_dict;
    phmap::flat_hash_map<node_type, Mapper<node_t, scalar_t>> mapper_dict;
    phmap::flat_hash_map<edge_type, NeighborSamplerImpl> sampler_dict;
    phmap::flat_hash_map<node_type, std::pair<size_t, size_t>> slice_dict;
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
            // sampler.uniform_sample(/*global_src_node=*/src_sampled_nodes[i],
            //                        /*local_src_node=*/i, count, dst_mapper,
            //                        generator, dst_sampled_nodes);
          }
        } else if constexpr (!std::is_scalar<node_t>::value) {  // Temporal:
          const at::Tensor& dst_time = time_dict.value().at(dst);
          const auto dst_time_data = dst_time.data_ptr<scalar_t>();
          for (size_t i = begin; i < end; ++i) {
            batch_id = src_sampled_nodes[i].first;
            // sampler.temporal_sample(/*global_src_node=*/src_sampled_nodes[i],
            //                         /*local_src_node=*/i, count,
            //                         seed_times[batch_id], dst_time_data,
            //                         dst_mapper, generator,
            //                         dst_sampled_nodes);
          }
        }
      }
      for (const auto& k : node_types) {
        slice_dict[k] = {slice_dict.at(k).second,
                         sampled_nodes_dict.at(k).size()};
      }
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

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  // TODO (matthias) fix automatic dispatching
  m.def(TORCH_SELECTIVE_NAME("pyg::hetero_neighbor_sample_cpu"),
        TORCH_FN(hetero_neighbor_sample_kernel));
}

}  // namespace sampler
}  // namespace pyg
