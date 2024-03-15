#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <omp.h>
#include <torch/library.h>
#include <iostream>
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
          bool save_edge_ids,
          bool distributed>
class NeighborSampler {
 public:
  NeighborSampler(const scalar_t* rowptr,
                  const scalar_t* col,
                  const std::string temporal_strategy)
      : rowptr_(rowptr), col_(col), temporal_strategy_(temporal_strategy) {
    TORCH_CHECK(temporal_strategy == "uniform" || temporal_strategy == "last",
                "No valid temporal strategy found");
  }

  void biased_sample(const node_t global_src_node,
                     const scalar_t local_src_node,
                     const at::Tensor& edge_weight,
                     const int64_t count,
                     pyg::sampler::Mapper<node_t, scalar_t>& dst_mapper,
                     pyg::random::RandintEngine<scalar_t>& generator,
                     std::vector<node_t>& out_global_dst_nodes) {
    const auto row_start = rowptr_[to_scalar_t(global_src_node)];
    const auto row_end = rowptr_[to_scalar_t(global_src_node) + 1];

    if ((row_end - row_start == 0) || (count == 0))
      return;

    const auto weight = edge_weight.narrow(0, row_start, row_end - row_start);

    _biased_sample(global_src_node, local_src_node, row_start, row_end, count,
                   weight, dst_mapper, generator, out_global_dst_nodes);
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

    if ((row_end - row_start == 0) || (count == 0))
      return;

    _sample(global_src_node, local_src_node, row_start, row_end, count,
            dst_mapper, generator, out_global_dst_nodes, node_counter);
  }

  void temporal_sample(const node_t global_src_node,
                       const scalar_t local_src_node,
                       const int64_t count,
                       const temporal_t seed_time,
                       const temporal_t* time,
                       pyg::sampler::Mapper<node_t, scalar_t>& dst_mapper,
                       pyg::random::RandintEngine<scalar_t>& generator,
                       std::vector<node_t>& out_global_dst_nodes) {
    auto row_start = rowptr_[to_scalar_t(global_src_node)];
    auto row_end = rowptr_[to_scalar_t(global_src_node) + 1];

    if ((row_end - row_start == 0) || (count == 0))
      return;

    // Find new `row_end` such that all neighbors fulfill temporal constraints:
    auto it = std::upper_bound(
        col_ + row_start, col_ + row_end, seed_time,
        [&](const scalar_t& a, const scalar_t& b) { return a < time[b]; });
    row_end = it - col_;

    if (temporal_strategy_ == "last" && count >= 0) {
      row_start = std::max(row_start, (scalar_t)(row_end - count));
    }

    if (row_end - row_start > 1) {
      TORCH_CHECK(time[col_[row_start]] <= time[col_[row_end - 1]],
                  "Found invalid non-sorted temporal neighborhood");
    }

    _sample(global_src_node, local_src_node, row_start, row_end, count,
            dst_mapper, generator, out_global_dst_nodes);
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

  // // base (Robert's implementation)
  void concat_results_base(
      std::vector<std::vector<int64_t>>& sampled_edges_size,
      int64_t total_num_edges) {
    _concat_results(sampled_rows_vec_, sampled_rows_, sampled_edges_size,
                    total_num_edges);
    _concat_results(sampled_cols_vec_, sampled_cols_, sampled_edges_size,
                    total_num_edges);
    if (save_edges) {
      _concat_results(sampled_edge_ids_vec_, sampled_edge_ids_,
                      sampled_edges_size, total_num_edges);
    }
  }

  // base (Robert's implementation)
  void update_sampled_edges_base(
      std::vector<std::vector<int64_t>>& num_sampled_nodes,
      std::vector<std::vector<int64_t>>& increment_values) {
    for (int64_t subgraph_idx = 0; subgraph_idx < num_sampled_nodes.size();
         ++subgraph_idx) {
      for (int64_t edge_idx = 0;
           edge_idx < sampled_cols_vec_[subgraph_idx].size(); ++edge_idx) {
        int64_t inc_col_idx = 0;
        for (int64_t inc_idx = increment_values[subgraph_idx].size() - 2;
             inc_idx >= 0; --inc_idx) {
          if (sampled_cols_vec_[subgraph_idx][edge_idx] >=
              num_sampled_nodes[subgraph_idx][inc_idx]) {
            inc_col_idx = inc_idx + 1;
            break;
          }
        }
        sampled_cols_vec_[subgraph_idx][edge_idx] +=
            increment_values[subgraph_idx][inc_col_idx];

        int64_t inc_row_idx = 0;
        for (int64_t inc_idx = increment_values[subgraph_idx].size() - 2;
             inc_idx >= 0; --inc_idx) {
          if (sampled_rows_vec_[subgraph_idx][edge_idx] >=
              num_sampled_nodes[subgraph_idx][inc_idx]) {
            inc_row_idx = inc_idx + 1;
            break;
          }
        }
        sampled_rows_vec_[subgraph_idx][edge_idx] +=
            increment_values[subgraph_idx][inc_row_idx];
      }
    }
  }

  // BEST OPTION 1
  void concat_results_1(std::vector<std::vector<int64_t>>& sampled_edges_size,
                        int64_t total_num_edges) {
    _concat_results_1(sampled_rows_vec_, sampled_cols_vec_,
                      sampled_edge_ids_vec_, sampled_rows_, sampled_cols_,
                      sampled_edge_ids_, sampled_edges_size, total_num_edges);
  }

  // BEST OPTION 1
  void update_sampled_edges_1(
      std::vector<std::vector<int64_t>>& num_sampled_nodes,
      std::vector<std::vector<int64_t>>& increment_values) {
    auto chunk_size = num_sampled_nodes.size() / at::get_num_threads();
    at::parallel_for(
        0, num_sampled_nodes.size(), chunk_size, [&](size_t _s, size_t _e) {
          for (int64_t subgraph_idx = _s; subgraph_idx < _e; ++subgraph_idx) {
            for (int64_t edge_idx = 0;
                 edge_idx < sampled_cols_vec_[subgraph_idx].size();
                 ++edge_idx) {
              int64_t inc_col_idx = 0;
              int64_t inc_row_idx = 0;
              for (int64_t inc_idx = increment_values[subgraph_idx].size() - 2;
                   inc_idx >= 0; --inc_idx) {
                if (inc_col_idx == 0 &&
                    sampled_cols_vec_[subgraph_idx][edge_idx] >=
                        num_sampled_nodes[subgraph_idx][inc_idx]) {
                  inc_col_idx = inc_idx + 1;
                }
                if (inc_row_idx == 0 &&
                    sampled_rows_vec_[subgraph_idx][edge_idx] >=
                        num_sampled_nodes[subgraph_idx][inc_idx]) {
                  inc_row_idx = inc_idx + 1;
                }
                if (inc_col_idx && inc_row_idx)
                  break;
              }

              sampled_cols_vec_[subgraph_idx][edge_idx] +=
                  increment_values[subgraph_idx][inc_col_idx];
              sampled_rows_vec_[subgraph_idx][edge_idx] +=
                  increment_values[subgraph_idx][inc_row_idx];
            }
          }
        });
  }

  // BEST OPTION 2
  void concat_and_update_edges_2(
      std::vector<std::vector<int64_t>>& sampled_edges_size,
      std::vector<std::vector<int64_t>>& num_sampled_nodes,
      std::vector<std::vector<int64_t>>& increment_values,
      int64_t total_num_edges) {
    _concat_and_update_edges_2(
        sampled_rows_vec_, sampled_cols_vec_, sampled_edge_ids_vec_,
        sampled_rows_, sampled_cols_, sampled_edge_ids_, sampled_edges_size,
        num_sampled_nodes, increment_values, total_num_edges);
  }

  // BEST OPTION 2
  void _concat_and_update_edges_2(
      std::vector<std::vector<scalar_t>>& sampled_rows_vec,
      std::vector<std::vector<scalar_t>>& sampled_cols_vec,
      std::vector<std::vector<scalar_t>>& sampled_edge_ids_vec,
      std::vector<scalar_t>& sampled_rows,
      std::vector<scalar_t>& sampled_cols,
      std::vector<scalar_t>& sampled_edge_ids,
      std::vector<std::vector<int64_t>>& sampled_edges_size,
      std::vector<std::vector<int64_t>>& num_sampled_nodes,
      std::vector<std::vector<int64_t>>& increment_values,
      int64_t total_num_edges) {
    sampled_rows.resize(total_num_edges);
    sampled_cols.resize(total_num_edges);
    if constexpr (save_edge_ids) {
      sampled_edge_ids.resize(total_num_edges);
    }

    int idx = 0;
    for (size_t ell = 0; ell < sampled_edges_size[0].size(); ++ell) {
      for (auto subgraph_idx = 0; subgraph_idx < sampled_edges_size.size();
           ++subgraph_idx) {
        int64_t edge_idx =
            ell > 0 ? sampled_edges_size[subgraph_idx][ell - 1] : 0;
        int64_t end = sampled_edges_size[subgraph_idx][ell];
        auto range = end - edge_idx;
        for (auto j = 0; j < range; j++) {
          int64_t inc_col_idx = 0;
          for (int64_t inc_idx = increment_values[subgraph_idx].size() - 2;
               inc_idx >= 0; --inc_idx) {
            if (sampled_cols_vec_[subgraph_idx][edge_idx] >=
                num_sampled_nodes[subgraph_idx][inc_idx]) {
              inc_col_idx = inc_idx + 1;
              break;
            }
          }
          sampled_cols[idx] = sampled_cols_vec_[subgraph_idx][edge_idx] +
                              increment_values[subgraph_idx][inc_col_idx];

          int64_t inc_row_idx = 0;
          for (int64_t inc_idx = increment_values[subgraph_idx].size() - 2;
               inc_idx >= 0; --inc_idx) {
            if (sampled_rows_vec_[subgraph_idx][edge_idx] >=
                num_sampled_nodes[subgraph_idx][inc_idx]) {
              inc_row_idx = inc_idx + 1;
              break;
            }
          }
          sampled_rows[idx] = sampled_rows_vec_[subgraph_idx][edge_idx] +
                              increment_values[subgraph_idx][inc_row_idx];

          if (save_edge_ids) {
            sampled_edge_ids[idx] =
                sampled_edge_ids_vec_[subgraph_idx][edge_idx];
          }

          edge_idx += 1;
          idx += 1;
        }
      }
    }
  }

  // with #pragma omp simd, layer parallel, layer parallel + simd reduction
  void _concat_results_merged_layer_parallel_and_simd(
      std::vector<std::vector<scalar_t>>& sampled_rows_vec,
      std::vector<std::vector<scalar_t>>& sampled_cols_vec,
      std::vector<std::vector<scalar_t>>& sampled_edge_ids_vec,
      std::vector<scalar_t>& sampled_rows,
      std::vector<scalar_t>& sampled_cols,
      std::vector<scalar_t>& sampled_edge_ids,
      std::vector<std::vector<int64_t>>& sampled_edges_size,
      std::vector<std::vector<int64_t>>& num_sampled_nodes,
      std::vector<std::vector<int64_t>>& increment_values,
      int64_t total_num_edges) {
    sampled_rows.resize(total_num_edges);
    sampled_cols.resize(total_num_edges);
    sampled_edge_ids.resize(total_num_edges);

    std::vector<int64_t> parallel_ranges = {0};

    for (auto i = 0; i < sampled_edges_size[0].size() - 1; ++i) {
      int64_t begin_range = 0;
#pragma omp simd reduction(+ : begin_range)
      for (auto j = 0; j < sampled_edges_size.size(); ++j) {
        begin_range += sampled_edges_size[j][i];
      }
      parallel_ranges.push_back(begin_range);
    }

    at::parallel_for(
        0, sampled_edges_size[0].size(), 1, [&](size_t _s, size_t _e) {
          for (size_t ell = _s; ell < _e; ++ell) {
            int idx = parallel_ranges[ell];
            for (auto subgraph_idx = 0;
                 subgraph_idx < sampled_edges_size.size(); ++subgraph_idx) {
              int64_t edge_idx =
                  ell > 0 ? sampled_edges_size[subgraph_idx][ell - 1] : 0;
              int64_t end = sampled_edges_size[subgraph_idx][ell];
              auto range = end - edge_idx;
              for (auto j = 0; j < range; ++j) {
                int64_t inc_col_idx = 0;
                for (int64_t inc_idx =
                         increment_values[subgraph_idx].size() - 2;
                     inc_idx >= 0; --inc_idx) {
                  if (sampled_cols_vec_[subgraph_idx][edge_idx] >=
                      num_sampled_nodes[subgraph_idx][inc_idx]) {
                    inc_col_idx = inc_idx + 1;
                    break;
                  }
                }
                sampled_cols[idx] = sampled_cols_vec_[subgraph_idx][edge_idx] +
                                    increment_values[subgraph_idx][inc_col_idx];

                int64_t inc_row_idx = 0;
                for (int64_t inc_idx =
                         increment_values[subgraph_idx].size() - 2;
                     inc_idx >= 0; --inc_idx) {
                  if (sampled_rows_vec_[subgraph_idx][edge_idx] >=
                      num_sampled_nodes[subgraph_idx][inc_idx]) {
                    inc_row_idx = inc_idx + 1;
                    break;
                  }
                }
                sampled_rows[idx] = sampled_rows_vec_[subgraph_idx][edge_idx] +
                                    increment_values[subgraph_idx][inc_row_idx];

                if (save_edge_ids) {
                  sampled_edge_ids[idx] =
                      sampled_edge_ids_vec_[subgraph_idx][edge_idx];
                }

                edge_idx += 1;
                idx += 1;
              }
            }
          }
        });
  }

  // Comment - below solution is 2 times (or more) slower !!
  void _concat_results_merged_writing_directly_to_tensors(
      std::vector<std::vector<int64_t>>& num_sampled_nodes,
      std::vector<std::vector<int64_t>>& increment_values,
      at::Tensor& out_row,
      at::Tensor& out_col,
      at::Tensor& out_node_id,
      c10::optional<at::Tensor>& out_edge_id,
      int64_t total_num_edges) {
    // sampled_rows.resize(total_num_edges);
    // sampled_cols.resize(total_num_edges);
    // sampled_edge_ids.resize(total_num_edges);

    out_row =
        at::empty({total_num_edges}, c10::CppTypeToScalarType<scalar_t>::value);
    out_col =
        at::empty({total_num_edges}, c10::CppTypeToScalarType<scalar_t>::value);

    if (save_edge_ids) {
      out_edge_id = at::empty({total_num_edges},
                              c10::CppTypeToScalarType<scalar_t>::value);
    }
    scalar_t* out_edge_id_data =
        save_edge_ids ? out_edge_id.value().data_ptr<scalar_t>() : nullptr;

    // std::chrono::steady_clock::time_point begin =
    std::chrono::steady_clock::now();
    int idx = 0;
    for (size_t ell = 0; ell < sampled_edges_size[0].size(); ++ell) {
      for (auto subgraph_idx = 0; subgraph_idx < sampled_edges_size.size();
           ++subgraph_idx) {
        int64_t edge_idx =
            ell > 0 ? sampled_edges_size[subgraph_idx][ell - 1] : 0;
        int64_t end = sampled_edges_size[subgraph_idx][ell];
        auto range = end - edge_idx;
        for (auto j = 0; j < range; j++) {
          int64_t inc_col_idx = 0;
          for (int64_t inc_idx = increment_values[subgraph_idx].size() - 2;
               inc_idx >= 0; --inc_idx) {
            if (sampled_cols_vec_[subgraph_idx][edge_idx] >=
                num_sampled_nodes[subgraph_idx][inc_idx]) {
              inc_col_idx = inc_idx + 1;
              break;
            }
          }
          out_col[idx] = sampled_cols_vec_[subgraph_idx][edge_idx] +
                         increment_values[subgraph_idx][inc_col_idx];
          // std::cout<<"out_col="<<out_col<<std::endl;

          int64_t inc_row_idx = 0;
          for (int64_t inc_idx = increment_values[subgraph_idx].size() - 2;
               inc_idx >= 0; --inc_idx) {
            if (sampled_rows_vec_[subgraph_idx][edge_idx] >=
                num_sampled_nodes[subgraph_idx][inc_idx]) {
              inc_row_idx = inc_idx + 1;
              break;
            }
          }
          out_row[idx] = sampled_rows_vec_[subgraph_idx][edge_idx] +
                         increment_values[subgraph_idx][inc_row_idx];
          // std::cout<<"out_row="<<out_row<<std::endl;

          if (save_edge_ids) {
            out_edge_id_data[idx] =
                sampled_edge_ids_vec_[subgraph_idx][edge_idx];
            // std::cout<<"out_edge_id="<<out_edge_id.value()<<std::endl;
          }

          // sampled_cols[idx] = sampled_cols_vec[subgraph_idx][edge_idx];
          // sampled_rows[idx] = sampled_rows_vec[subgraph_idx][edge_idx];
          edge_idx += 1;
          idx += 1;
        }
      }
    }
    // std::chrono::steady_clock::time_point end =
    //   std::chrono::steady_clock::now();
    // std::cout<<"time="<<(end - begin)<<std::endl;
    //   std::chrono::steady_clock::time_point end =
    std::chrono::steady_clock::now();
    // std::cout << "Total time: " <<
    std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
        << std::endl;
  }

 private:
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

  // BEST OPTION 1
  void _concat_results_1(
      std::vector<std::vector<scalar_t>>& sampled_rows_vec,
      std::vector<std::vector<scalar_t>>& sampled_cols_vec,
      std::vector<std::vector<scalar_t>>& sampled_edge_ids_vec,
      std::vector<scalar_t>& sampled_rows,
      std::vector<scalar_t>& sampled_cols,
      std::vector<scalar_t>& sampled_edge_ids,
      std::vector<std::vector<int64_t>>& sampled_edges_size,
      int64_t total_num_edges) {
    sampled_rows.reserve(total_num_edges + 1);
    sampled_cols.reserve(total_num_edges + 1);
    if constexpr (save_edges) {
      sampled_edge_ids.reserve(total_num_edges + 1);
    }
    for (size_t ell = 0; ell < sampled_edges_size[0].size(); ++ell) {
      for (auto i = 0; i < sampled_edges_size.size(); ++i) {
        int64_t beg = ell > 0 ? sampled_edges_size[i][ell - 1] : 0;
        int64_t end = sampled_edges_size[i][ell];
        std::copy(sampled_rows_vec[i].begin() + beg,
                  sampled_rows_vec[i].begin() + end,
                  std::back_inserter(sampled_rows));
        std::copy(sampled_cols_vec[i].begin() + beg,
                  sampled_cols_vec[i].begin() + end,
                  std::back_inserter(sampled_cols));
        if constexpr (save_edges) {
          std::copy(sampled_edge_ids_vec[i].begin() + beg,
                    sampled_edge_ids_vec[i].begin() + end,
                    std::back_inserter(sampled_edge_ids));
        }
      }
    }
  }

  // update_edges and concat merged - edge ids simd
  void _concat_edge_ids_(
      std::vector<std::vector<scalar_t>>& sampled_edge_ids_vec,
      std::vector<scalar_t>& sampled_edge_ids,
      std::vector<std::vector<int64_t>>& sampled_edges_size,
      int64_t total_num_edges) {
    sampled_edge_ids.resize(total_num_edges);
    int idx = 0;
    for (size_t ell = 0; ell < sampled_edges_size[0].size(); ++ell) {
      for (auto i = 0; i < sampled_edges_size.size(); ++i) {
        int64_t shift = ell > 0 ? sampled_edges_size[i][ell - 1] : 0;
        int64_t end = sampled_edges_size[i][ell];
        auto range = end - shift;
#pragma omp simd
        for (auto j = 0; j < range; j++) {
          sampled_edge_ids[idx] = sampled_edge_ids_vec[i][shift];
        }
        shift += 1;
        idx += 1;
      }
    }
  }

  // base (Robert's implementation)
  void _concat_results_base(
      std::vector<std::vector<scalar_t>>& sampled_edges_vec,
      std::vector<scalar_t>& sampled_edges,
      std::vector<std::vector<int64_t>>& sampled_edges_size,
      int64_t total_num_edges) {
    sampled_edges.reserve(total_num_edges + 1);
    for (size_t ell = 0; ell < sampled_edges_size[0].size(); ++ell) {
      for (auto i = 0; i < sampled_edges_size.size(); ++i) {
        int64_t beg = ell > 0 ? sampled_edges_size[i][ell - 1] : 0;
        int64_t end = sampled_edges_size[i][ell];
        std::copy(sampled_edges_vec[i].begin() + beg,
                  sampled_edges_vec[i].begin() + end,
                  std::back_inserter(sampled_edges));
      }
    }
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
    const auto population = row_end - row_start;

    // Case 1: Sample the full neighborhood:
    if (count < 0 || (!replace && count >= population)) {
      for (scalar_t edge_id = row_start; edge_id < row_end; ++edge_id) {
        add(edge_id, global_src_node, local_src_node, dst_mapper,
            out_global_dst_nodes, node_counter);
      }
    }

    // Case 2: Sample with replacement:
    else if (replace) {
      if (row_end < (1 << 16)) {
        const auto arr = std::move(
            generator.generate_range_of_ints(row_start, row_end, count));
        for (const auto edge_id : arr)
          add(edge_id, global_src_node, local_src_node, dst_mapper,
              out_global_dst_nodes, node_counter);
      } else {
        for (int64_t i = 0; i < count; ++i) {
          const auto edge_id = generator(row_start, row_end);
          add(edge_id, global_src_node, local_src_node, dst_mapper,
              out_global_dst_nodes, node_counter);
        }
      }
    }

    // Case 3: Sample without replacement:
    else {
      auto index_tracker = IndexTracker<scalar_t>(population);
      if (population < (1 << 16)) {
        const auto arr =
            std::move(generator.generate_range_of_ints(0, population, count));
        for (auto i = 0; i < arr.size(); ++i) {
          auto rnd = arr[i];
          if (!index_tracker.try_insert(rnd)) {
            rnd = population - count + i;
            index_tracker.insert(population - count + i);
          }
          const auto edge_id = row_start + rnd;
          add(edge_id, global_src_node, local_src_node, dst_mapper,
              out_global_dst_nodes, node_counter);
        }
      } else {
        for (auto i = population - count; i < population; ++i) {
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
  }

  void _biased_sample(const node_t global_src_node,
                      const scalar_t local_src_node,
                      const scalar_t row_start,
                      const scalar_t row_end,
                      const int64_t count,
                      const at::Tensor& weight,
                      pyg::sampler::Mapper<node_t, scalar_t>& dst_mapper,
                      pyg::random::RandintEngine<scalar_t>& generator,
                      std::vector<node_t>& out_global_dst_nodes) {
    const auto population = row_end - row_start;

    // Case 1: Sample the full neighborhood:
    if (count < 0 || (!replace && count >= population)) {
      for (scalar_t edge_id = row_start; edge_id < row_end; ++edge_id) {
        add(edge_id, global_src_node, local_src_node, dst_mapper,
            out_global_dst_nodes);
      }
    }

    // Case 2: Multinomial sampling:
    else {
      const auto index = at::multinomial(weight, count, replace);
      const auto index_data = index.data_ptr<int64_t>();
      for (size_t i = 0; i < index.numel(); ++i) {
        add(row_start + index_data[i], global_src_node, local_src_node,
            dst_mapper, out_global_dst_nodes);
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

    // In the distributed sampling case, we do not perform any mapping:
    if constexpr (distributed) {
      out_global_dst_nodes.push_back(global_dst_node);
      if (save_edge_ids) {
        sampled_edge_ids_.push_back(edge_id);
      }
      return;
    }

    const auto res = dst_mapper.insert(global_dst_node);
    if (res.second) {  // not yet sampled.
      out_global_dst_nodes.push_back(global_dst_node);
    }
    const bool parallel = node_counter != nullptr;
    if (!parallel) {
      // std::cout << "not parallel" << std::endl;
      if (save_edges) {
        num_sampled_edges_per_hop[num_sampled_edges_per_hop.size() - 1]++;
        sampled_rows_.push_back(local_src_node);
        sampled_cols_.push_back(res.first);
        if (save_edge_ids) {
          sampled_edge_ids_.push_back(edge_id);
        }
      }
    } else {
      // std::cout << "parallel/" << std::endl;
      if (save_edges) {
        if constexpr (!std::is_scalar<node_t>::value) {
          // num_sampled_edges_per_hop[num_sampled_edges_per_hop.size() - 1]++;
          sampled_rows_vec_[*node_counter].push_back(local_src_node);
          sampled_cols_vec_[*node_counter].push_back(res.first);

          if (save_edge_ids) {
            sampled_edge_ids_vec_[*node_counter].push_back(edge_id);
          }
        }
      }
    }
  }

  const scalar_t* rowptr_;
  const scalar_t* col_;
  const std::string temporal_strategy_;
  std::vector<scalar_t> sampled_rows_;
  std::vector<scalar_t> sampled_cols_;
  std::vector<scalar_t> sampled_edge_ids_;

 public:
  std::vector<std::vector<scalar_t>> sampled_rows_vec_;
  std::vector<std::vector<scalar_t>> sampled_cols_vec_;
  std::vector<std::vector<scalar_t>> sampled_edge_ids_vec_;
  std::vector<int64_t> num_sampled_edges_per_hop;
};

// Homogeneous neighbor sampling ///////////////////////////////////////////////

template <bool replace,
          bool directed,
          bool disjoint,
          bool return_edge_id,
          bool distributed>
std::tuple<at::Tensor,
           at::Tensor,
           at::Tensor,
           c10::optional<at::Tensor>,
           std::vector<int64_t>,
           std::vector<int64_t>,
           std::vector<int64_t>>
sample_old(const at::Tensor& rowptr,
           const at::Tensor& col,
           const at::Tensor& seed,
           const std::vector<int64_t>& num_neighbors,
           const c10::optional<at::Tensor>& node_time,
           const c10::optional<at::Tensor>& seed_time,
           const c10::optional<at::Tensor>& edge_weight,
           const bool csc,
           const std::string temporal_strategy) {
  TORCH_CHECK(!node_time.has_value() || disjoint,
              "Temporal sampling needs to create disjoint subgraphs");

  TORCH_CHECK(rowptr.is_contiguous(), "Non-contiguous 'rowptr'");
  TORCH_CHECK(col.is_contiguous(), "Non-contiguous 'col'");
  TORCH_CHECK(seed.is_contiguous(), "Non-contiguous 'seed'");
  if (node_time.has_value()) {
    TORCH_CHECK(node_time.value().is_contiguous(),
                "Non-contiguous 'node_time'");
  }
  if (seed_time.has_value()) {
    TORCH_CHECK(seed_time.value().is_contiguous(),
                "Non-contiguous 'seed_time'");
  }
  TORCH_CHECK(!(node_time.has_value() && edge_weight.has_value()),
              "Biased node temporal sampling not yet supported");

  at::Tensor out_row, out_col, out_node_id;
  c10::optional<at::Tensor> out_edge_id = c10::nullopt;
  std::vector<int64_t> num_sampled_nodes_per_hop;
  std::vector<int64_t> num_sampled_edges_per_hop;
  std::vector<int64_t> cumsum_neighbors_per_node =
      distributed ? std::vector<int64_t>(1, seed.size(0))
                  : std::vector<int64_t>();

  AT_DISPATCH_INTEGRAL_TYPES(seed.scalar_type(), "sample_kernel", [&] {
    typedef std::pair<scalar_t, scalar_t> pair_scalar_t;
    typedef std::conditional_t<!disjoint, scalar_t, pair_scalar_t> node_t;
    // TODO(zeyuan): Do not force int64_t for time type.
    typedef int64_t temporal_t;
    typedef NeighborSampler<node_t, scalar_t, temporal_t, replace, directed,
                            return_edge_id, distributed>
        NeighborSamplerImpl;

    pyg::random::RandintEngine<scalar_t> generator;

    std::vector<node_t> sampled_nodes;
    auto mapper = Mapper<node_t, scalar_t>(/*num_nodes=*/rowptr.size(0) - 1);
    auto sampler =
        NeighborSamplerImpl(rowptr.data_ptr<scalar_t>(),
                            col.data_ptr<scalar_t>(), temporal_strategy);
    std::vector<temporal_t> seed_times;

    const auto seed_data = seed.data_ptr<scalar_t>();
    if constexpr (!disjoint) {
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
      } else if (node_time.has_value()) {
        const auto time_data = node_time.value().data_ptr<temporal_t>();
        for (size_t i = 0; i < seed.numel(); ++i) {
          seed_times.push_back(time_data[seed_data[i]]);
        }
      }
    }

    num_sampled_nodes_per_hop.push_back(seed.numel());

    size_t begin = 0, end = seed.size(0);
    for (size_t ell = 0; ell < num_neighbors.size(); ++ell) {
      const auto count = num_neighbors[ell];
      sampler.num_sampled_edges_per_hop.push_back(0);
      if (edge_weight.has_value()) {
        for (size_t i = begin; i < end; ++i) {
          sampler.biased_sample(
              /*global_src_node=*/sampled_nodes[i],
              /*local_src_node=*/i,
              /*edge_weight=*/edge_weight.value(),
              /*count=*/count,
              /*dst_mapper=*/mapper,
              /*generator=*/generator,
              /*out_global_dst_nodes=*/sampled_nodes);
          if constexpr (distributed)
            cumsum_neighbors_per_node.push_back(sampled_nodes.size());
        }
      } else if (!node_time.has_value()) {
        for (size_t i = begin; i < end; ++i) {
          sampler.uniform_sample(
              /*global_src_node=*/sampled_nodes[i],
              /*local_src_node=*/i,
              /*count=*/count,
              /*dst_mapper=*/mapper,
              /*generator=*/generator,
              /*out_global_dst_nodes=*/sampled_nodes);
          if constexpr (distributed)
            cumsum_neighbors_per_node.push_back(sampled_nodes.size());
        }
      }
      begin = end, end = sampled_nodes.size();
      num_sampled_nodes_per_hop.push_back(end - begin);
    }

    out_node_id = pyg::utils::from_vector(sampled_nodes);
    TORCH_CHECK(directed, "Undirected subgraphs not yet supported");
    if (directed) {
      std::tie(out_row, out_col, out_edge_id) = sampler.get_sampled_edges(csc);
    } else {
      TORCH_CHECK(!disjoint, "Disjoint subgraphs not yet supported");
    }

    num_sampled_edges_per_hop = sampler.num_sampled_edges_per_hop;
  });

  return std::make_tuple(out_row, out_col, out_node_id, out_edge_id,
                         num_sampled_nodes_per_hop, num_sampled_edges_per_hop,
                         cumsum_neighbors_per_node);
}

// Homogeneous neighbor sampling ///////////////////////////////////////////////

template <bool replace,
          bool directed,
          bool disjoint,
          bool return_edge_id,
          bool distributed>
std::tuple<at::Tensor,
           at::Tensor,
           at::Tensor,
           c10::optional<at::Tensor>,
           std::vector<int64_t>,
           std::vector<int64_t>,
           std::vector<int64_t>>
sample(const at::Tensor& rowptr,
       const at::Tensor& col,
       const at::Tensor& seed,
       const std::vector<int64_t>& num_neighbors,
       const c10::optional<at::Tensor>& time,
       const c10::optional<at::Tensor>& seed_time,
       const c10::optional<at::Tensor>& edge_weight,
       const bool csc,
       const std::string temporal_strategy,
       const bool old) {
  // if (old) {
  //   return sample_old<replace, directed, disjoint, return_edge_id,
  //   distributed>(
  //       rowptr, col, seed, num_neighbors, time, seed_time, edge_weight, csc,
  //       temporal_strategy);
  // }
  // std::chrono::steady_clock::time_point sampl_begin =
  //     std::chrono::steady_clock::now();
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
  TORCH_CHECK(!(time.has_value() && edge_weight.has_value()),
              "Biased temporal sampling not yet supported");

  at::Tensor out_row, out_col, out_node_id;
  c10::optional<at::Tensor> out_edge_id = c10::nullopt;
  std::vector<int64_t> num_sampled_nodes_per_hop(num_neighbors.size() + 1);
  std::vector<int64_t> num_sampled_edges_per_hop(num_neighbors.size());
  std::vector<int64_t> cumsum_neighbors_per_node =
      distributed ? std::vector<int64_t>(1, seed.size(0))
                  : std::vector<int64_t>();

  AT_DISPATCH_INTEGRAL_TYPES(seed.scalar_type(), "sample_kernel", [&] {
    typedef std::pair<scalar_t, scalar_t> pair_scalar_t;
    typedef std::conditional_t<!disjoint, scalar_t, pair_scalar_t> node_t;
    // TODO(zeyuan): Do not force int64_t for time type.
    typedef int64_t temporal_t;
    typedef NeighborSampler<node_t, scalar_t, temporal_t, replace, directed,
                            return_edge_id, distributed>
        NeighborSamplerImpl;
    // -------------------------------
    // SEMI-DISJOINT CODE start here
    // std::cout << "Semi-disjoint sampling starts" << std::endl;
    const auto seed_size = seed.size(0);

    std::vector<std::vector<int64_t>> num_nodes(seed_size);
    std::vector<std::vector<node_t>> subgraph_sampled_nodes(seed_size);
    std::vector<std::vector<int64_t>> sampled_edges_size(seed_size);
    std::vector<node_t> sampled_nodes;

    // auto mapper = Mapper<node_t, scalar_t>(rowptr.size(0) - 1);
    auto sampler =
        NeighborSamplerImpl(rowptr.data_ptr<scalar_t>(),
                            col.data_ptr<scalar_t>(), temporal_strategy);
    std::vector<temporal_t> seed_times;

    sampler.sampled_cols_vec_.resize(seed_size);
    sampler.sampled_rows_vec_.resize(seed_size);
    sampler.sampled_edge_ids_vec_.resize(seed_size);

    const auto seed_data = seed.data_ptr<scalar_t>();
    if constexpr (!disjoint) {
      // std::cout << "THIS POC IS ONLY FOR DISJOIT MODE" << std::endl;
      // sampled_nodes = pyg::utils::to_vector<scalar_t>(seed);
      // mapper.fill(seed);
    } else {
      if constexpr (!std::is_scalar<node_t>::value) {
        for (size_t i = 0; i < seed.numel(); ++i) {
          //	std::pair<scalar_t, scalar_t> seed_node{0, seed_data[i]};
          subgraph_sampled_nodes[i].push_back({i, seed_data[i]});
          sampled_nodes.push_back({i, seed_data[i]});
          // mapper.insert({i, seed_data[i]});
          num_nodes[i].push_back(1);
        }
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

    // std::cout<<"num_sampled_nodes_per_hop="<<num_sampled_nodes_per_hop<<std::endl;
    num_sampled_nodes_per_hop[0] = seed.numel();

    // size_t begin = 0, end = seed.size(0);
    //  std::cout<<"num_threads="<<at::get_num_threads()<<std::endl;
    auto chunk_size = seed_size / at::get_num_threads();
    // std::cout<<"chunk_size="<<chunk_size<<", seed_size="<<seed_size<<",
    // num_threads="<<at::get_num_threads()<<std::endl; std::cout << "Parallel
    // loop begin" << std::endl;

    at::parallel_for(0, seed_size, chunk_size, [&](size_t _s, size_t _e) {
      pyg::random::RandintEngine<scalar_t> generator;
      auto ell_size = num_neighbors.size();
      for (auto j = _s; j < _e; j++) {
        int64_t scope_begin(0);
        int node_counter = j;
        auto mapper =
            Mapper<node_t, scalar_t>(/*num_nodes=*/rowptr.size(0) - 1);
        if constexpr (!std::is_scalar<node_t>::value) {
          mapper.insert({j, seed_data[j]});
        }
        for (size_t ell = 0; ell < ell_size; ++ell) {
          ////std::cout << "Seed id: "<< j << " level: " << ell << std::endl;
          const auto count = num_neighbors[ell];
          // sampler.num_sampled_edges_per_hop.push_back(0);
          for (auto i = scope_begin; i < num_nodes[j].back(); ++i) {
            sampler.uniform_sample(
                /*global_src_node=*/subgraph_sampled_nodes[j][i],
                /*local_src_node=*/i,
                /*count=*/count,
                /*dst_mapper=*/mapper,
                /*generator=*/generator,
                /*out_global_dst_nodes=*/subgraph_sampled_nodes[j],
                /*node_counter=*/&node_counter);
          }
          scope_begin = num_nodes[j].back();
          num_nodes[j].push_back(subgraph_sampled_nodes[j].size());
          sampled_edges_size[j].push_back(sampler.sampled_cols_vec_[j].size());
          // num_sampled_nodes_per_hop.push_back(end - begin);
        }
      }
    });
    // std::cout << "Parallel loop end" << std::endl;

    // SEMI-DISJOINT POSTPROCESSING

    int64_t total_num_nodes = 0;
    int64_t total_num_edges = 0;
    for (size_t seed_idx = 0; seed_idx < seed_size; seed_idx++) {
      total_num_nodes += num_nodes[seed_idx].back();
      total_num_edges += sampled_edges_size[seed_idx].back();
    }

    // concatenate sampled nodes
    sampled_nodes.reserve(total_num_nodes + 1);
    for (size_t ell = 0; ell < num_neighbors.size(); ++ell) {
      for (size_t seed_idx = 0; seed_idx < seed_size; seed_idx++) {
        std::copy(
            subgraph_sampled_nodes[seed_idx].begin() + num_nodes[seed_idx][ell],
            subgraph_sampled_nodes[seed_idx].begin() +
                num_nodes[seed_idx][ell + 1],
            std::back_inserter(sampled_nodes));

        num_sampled_nodes_per_hop[ell + 1] +=
            num_nodes[seed_idx][ell + 1] - num_nodes[seed_idx][ell];

        num_sampled_edges_per_hop[ell] +=
            ell == 0 ? sampled_edges_size[seed_idx][ell]
                     : sampled_edges_size[seed_idx][ell] -
                           sampled_edges_size[seed_idx][ell - 1];
      }
    }
    // concatenate sampled_rows_, sampled_cols_ and sampled_edge_ids_
    std::vector<std::vector<int64_t>> increment_values(seed_size);
    int64_t increment_value = 0;
    for (size_t seed_idx = 0; seed_idx < seed_size; seed_idx++) {
      increment_values[seed_idx].push_back(increment_value);
      increment_value += num_nodes[seed_idx][0];
    }
    for (size_t ell = 1; ell <= num_neighbors.size(); ++ell) {
      for (size_t seed_idx = 0; seed_idx < seed_size; seed_idx++) {
        increment_value -= num_nodes[seed_idx][ell - 1];
        increment_values[seed_idx].push_back(increment_value);
        increment_value += num_nodes[seed_idx][ell];
      }
    }
    // std::a << "After creation of increment_vales" << std::endl;
    // sampler.update_edges(num_nodes, increment_values);
    // std::cout << "After updates of sampled col and row vec" << std::endl;

    sampler.concat_and_update_edges_2(
        sampled_edges_size, num_nodes, increment_values,
        total_num_edges);
    // std::cout << "Semi-disjoint sampler end" << std::endl;

    out_node_id = pyg::utils::from_vector(sampled_nodes);
    TORCH_CHECK(directed, "Undirected subgraphs not yet supported");
    if (directed) {
      std::tie(out_row, out_col, out_edge_id) = sampler.get_sampled_edges(csc);
    } else {
      TORCH_CHECK(!disjoint, "Disjoint subgraphs not yet supported");
    }
  });
  // std::chrono::steady_clock::time_point sampl_end =
  // std::chrono::steady_clock::now();
  ////std::cout << "Total time: " <<
  /// std::chrono::duration_cast<std::chrono::microseconds>(sampl_end -
  /// sampl_begin).count() << std::endl;

  return std::make_tuple(out_row, out_col, out_node_id, out_edge_id,
                         num_sampled_nodes_per_hop, num_sampled_edges_per_hop,
                         cumsum_neighbors_per_node);
}

// Heterogeneous neighbor sampling /////////////////////////////////////////////

template <bool replace,
          bool directed,
          bool disjoint,
          bool return_edge_id,
          bool distributed>
std::tuple<c10::Dict<rel_type, at::Tensor>,
           c10::Dict<rel_type, at::Tensor>,
           c10::Dict<node_type, at::Tensor>,
           c10::optional<c10::Dict<rel_type, at::Tensor>>,
           c10::Dict<node_type, std::vector<int64_t>>,
           c10::Dict<rel_type, std::vector<int64_t>>>
sample(const std::vector<node_type>& node_types,
       const std::vector<edge_type>& edge_types,
       const c10::Dict<rel_type, at::Tensor>& rowptr_dict,
       const c10::Dict<rel_type, at::Tensor>& col_dict,
       const c10::Dict<node_type, at::Tensor>& seed_dict,
       const c10::Dict<rel_type, std::vector<int64_t>>& num_neighbors_dict,
       const c10::optional<c10::Dict<node_type, at::Tensor>>& time_dict,
       const c10::optional<c10::Dict<node_type, at::Tensor>>& seed_time_dict,
       const c10::optional<c10::Dict<rel_type, at::Tensor>>& edge_weight_dict,
       const bool csc,
       const std::string temporal_strategy) {
  TORCH_CHECK(!time_dict.has_value() || disjoint,
              "Temporal sampling needs to create disjoint subgraphs");
  std::chrono::steady_clock::time_point sampl_begin =
      std::chrono::steady_clock::now();
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
  TORCH_CHECK(!(time_dict.has_value() && edge_weight_dict.has_value()),
              "Biased temporal sampling not yet supported");

  c10::Dict<rel_type, at::Tensor> out_row_dict, out_col_dict;
  c10::Dict<node_type, at::Tensor> out_node_id_dict;
  c10::optional<c10::Dict<node_type, at::Tensor>> out_edge_id_dict;
  if (return_edge_id) {
    out_edge_id_dict = c10::Dict<rel_type, at::Tensor>();
  } else {
    out_edge_id_dict = c10::nullopt;
  }
  std::unordered_map<node_type, std::vector<int64_t>>
      num_sampled_nodes_per_hop_map;
  c10::Dict<node_type, std::vector<int64_t>> num_sampled_nodes_per_hop_dict;
  c10::Dict<rel_type, std::vector<int64_t>> num_sampled_edges_per_hop_dict;

  const auto scalar_type = seed_dict.begin()->value().scalar_type();
  AT_DISPATCH_INTEGRAL_TYPES(scalar_type, "hetero_sample_kernel", [&] {
    typedef std::pair<scalar_t, scalar_t> pair_scalar_t;
    typedef std::conditional_t<!disjoint, scalar_t, pair_scalar_t> node_t;
    typedef int64_t temporal_t;
    typedef NeighborSampler<node_t, scalar_t, temporal_t, replace, directed,
                            return_edge_id, distributed>
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
      num_sampled_nodes_per_hop_map.insert({k, std::vector<int64_t>(1, 0)});
      mapper_dict.insert({k, Mapper<node_t, scalar_t>(N)});
      slice_dict[k] = {0, 0};
    }

    const bool parallel = at::get_num_threads() > 1 && edge_types.size() > 1;
    std::vector<std::vector<edge_type>> threads_edge_types;

    for (const auto& k : edge_types) {
      L = std::max(L, num_neighbors_dict.at(to_rel_type(k)).size());
      sampler_dict.insert(
          {k, NeighborSamplerImpl(
                  rowptr_dict.at(to_rel_type(k)).data_ptr<scalar_t>(),
                  col_dict.at(to_rel_type(k)).data_ptr<scalar_t>(),
                  temporal_strategy)});

      if (parallel) {
        // Each thread is assigned edge types that have the same dst node
        // type. Thanks to this, each thread will operate on a separate mapper
        // and separate sampler.
        bool added = false;
        const auto dst = !csc ? std::get<2>(k) : std::get<0>(k);
        for (auto& e : threads_edge_types) {
          if ((!csc ? std::get<2>(e[0]) : std::get<0>(e[0])) == dst) {
            e.push_back(k);
            added = true;
            break;
          }
        }
        if (!added)
          threads_edge_types.push_back({k});
      }
    }
    if (!parallel) {  // One thread handles all edge types.
      threads_edge_types.push_back({edge_types});
    }

    scalar_t batch_idx = 0;
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
          sampled_nodes.push_back({batch_idx, seed_data[i]});
          mapper.insert({batch_idx, seed_data[i]});
          batch_idx++;
        }
        if (seed_time_dict.has_value()) {
          const at::Tensor& seed_time = seed_time_dict.value().at(kv.key());
          const auto seed_time_data = seed_time.data_ptr<scalar_t>();
          seed_times.reserve(seed_times.size() + seed.numel());
          for (size_t i = 0; i < seed.numel(); ++i) {
            seed_times.push_back(seed_time_data[i]);
          }
        } else if (time_dict.has_value()) {
          const at::Tensor& time = time_dict.value().at(kv.key());
          const auto time_data = time.data_ptr<scalar_t>();
          seed_times.reserve(seed_times.size() + seed.numel());
          for (size_t i = 0; i < seed.numel(); ++i) {
            seed_times.push_back(time_data[seed_data[i]]);
          }
        }
      }

      num_sampled_nodes_per_hop_map.at(kv.key())[0] =
          sampled_nodes_dict.at(kv.key()).size();
    }

    size_t begin, end;
    for (size_t ell = 0; ell < L; ++ell) {
      phmap::flat_hash_map<node_type, std::vector<node_t>>
          dst_sampled_nodes_dict;
      if (parallel) {
        for (const auto& k : threads_edge_types) {  // Intialize empty vectors.
          dst_sampled_nodes_dict[!csc ? std::get<2>(k[0]) : std::get<0>(k[0])];
        }
      }
      at::parallel_for(
          0, threads_edge_types.size(), 1, [&](size_t _s, size_t _e) {
            for (auto j = _s; j < _e; ++j) {
              for (const auto& k : threads_edge_types[j]) {
                const auto src = !csc ? std::get<0>(k) : std::get<2>(k);
                const auto dst = !csc ? std::get<2>(k) : std::get<0>(k);
                const auto count = num_neighbors_dict.at(to_rel_type(k))[ell];
                auto& src_sampled_nodes = sampled_nodes_dict.at(src);
                auto& dst_sampled_nodes = parallel
                                              ? dst_sampled_nodes_dict.at(dst)
                                              : sampled_nodes_dict.at(dst);
                auto& dst_mapper = mapper_dict.at(dst);
                auto& sampler = sampler_dict.at(k);
                size_t begin, end;
                std::tie(begin, end) = slice_dict.at(src);

                sampler.num_sampled_edges_per_hop.push_back(0);

                if (edge_weight_dict.has_value() &&
                    edge_weight_dict.value().contains(to_rel_type(k))) {
                  const at::Tensor& edge_weight =
                      edge_weight_dict.value().at(to_rel_type(k));
                  for (size_t i = begin; i < end; ++i) {
                    sampler.biased_sample(
                        /*global_src_node=*/src_sampled_nodes[i],
                        /*local_src_node=*/i,
                        /*edge_weight=*/edge_weight,
                        /*count=*/count,
                        /*dst_mapper=*/dst_mapper,
                        /*generator=*/generator,
                        /*out_global_dst_nodes=*/dst_sampled_nodes);
                  }
                } else if (!time_dict.has_value() ||
                           !time_dict.value().contains(dst)) {
                  for (size_t i = begin; i < end; ++i) {
                    sampler.uniform_sample(
                        /*global_src_node=*/src_sampled_nodes[i],
                        /*local_src_node=*/i,
                        /*count=*/count,
                        /*dst_mapper=*/dst_mapper,
                        /*generator=*/generator,
                        /*out_global_dst_nodes=*/dst_sampled_nodes);
                  }
                } else if constexpr (!std::is_scalar<node_t>::value) {
                  // Temporal sampling:
                  const at::Tensor& dst_time = time_dict.value().at(dst);
                  const auto dst_time_data = dst_time.data_ptr<temporal_t>();
                  for (size_t i = begin; i < end; ++i) {
                    const auto batch_idx = src_sampled_nodes[i].first;
                    sampler.temporal_sample(
                        /*global_src_node=*/src_sampled_nodes[i],
                        /*local_src_node=*/i,
                        /*count=*/count,
                        /*seed_time=*/seed_times[batch_idx],
                        /*time=*/dst_time_data,
                        /*dst_mapper=*/dst_mapper,
                        /*generator=*/generator,
                        /*out_global_dst_nodes=*/dst_sampled_nodes);
                  }
                }
              }
            }
          });

      if (parallel) {
        for (auto& dst_sampled : dst_sampled_nodes_dict) {
          std::copy(dst_sampled.second.begin(), dst_sampled.second.end(),
                    std::back_inserter(sampled_nodes_dict[dst_sampled.first]));
        }
      }
      for (const auto& k : node_types) {
        slice_dict[k] = {slice_dict.at(k).second,
                         sampled_nodes_dict.at(k).size()};
        num_sampled_nodes_per_hop_map.at(k).push_back(slice_dict.at(k).second -
                                                      slice_dict.at(k).first);
      }
    }

    for (const auto& k : node_types) {
      out_node_id_dict.insert(
          k, pyg::utils::from_vector(sampled_nodes_dict.at(k)));
      num_sampled_nodes_per_hop_dict.insert(
          k, num_sampled_nodes_per_hop_map.at(k));
    }

    TORCH_CHECK(directed, "Undirected heterogeneous graphs not yet supported");
    if (directed) {
      for (const auto& k : edge_types) {
        const auto edges = sampler_dict.at(k).get_sampled_edges(csc);
        out_row_dict.insert(to_rel_type(k), std::get<0>(edges));
        out_col_dict.insert(to_rel_type(k), std::get<1>(edges));
        num_sampled_edges_per_hop_dict.insert(
            to_rel_type(k), sampler_dict.at(k).num_sampled_edges_per_hop);
        if (return_edge_id) {
          out_edge_id_dict.value().insert(to_rel_type(k),
                                          std::get<2>(edges).value());
        }
      }
    }
  });

  return std::make_tuple(out_row_dict, out_col_dict, out_node_id_dict,
                         out_edge_id_dict, num_sampled_nodes_per_hop_dict,
                         num_sampled_edges_per_hop_dict);
}

// Dispatcher //////////////////////////////////////////////////////////////////

#define DISPATCH_SAMPLE(replace, directed, disjoint, return_edge_id, ...) \
  if (replace && directed && disjoint && return_edge_id)                  \
    return sample<true, true, true, true, false>(__VA_ARGS__);            \
  if (replace && directed && disjoint && !return_edge_id)                 \
    return sample<true, true, true, false, false>(__VA_ARGS__);           \
  if (replace && directed && !disjoint && return_edge_id)                 \
    return sample<true, true, false, true, false>(__VA_ARGS__);           \
  if (replace && directed && !disjoint && !return_edge_id)                \
    return sample<true, true, false, false, false>(__VA_ARGS__);          \
  if (replace && !directed && disjoint && return_edge_id)                 \
    return sample<true, false, true, true, false>(__VA_ARGS__);           \
  if (replace && !directed && disjoint && !return_edge_id)                \
    return sample<true, false, true, false, false>(__VA_ARGS__);          \
  if (replace && !directed && !disjoint && return_edge_id)                \
    return sample<true, false, false, true, false>(__VA_ARGS__);          \
  if (replace && !directed && !disjoint && !return_edge_id)               \
    return sample<true, false, false, false, false>(__VA_ARGS__);         \
  if (!replace && directed && disjoint && return_edge_id)                 \
    return sample<false, true, true, true, false>(__VA_ARGS__);           \
  if (!replace && directed && disjoint && !return_edge_id)                \
    return sample<false, true, true, false, false>(__VA_ARGS__);          \
  if (!replace && directed && !disjoint && return_edge_id)                \
    return sample<false, true, false, true, false>(__VA_ARGS__);          \
  if (!replace && directed && !disjoint && !return_edge_id)               \
    return sample<false, true, false, false, false>(__VA_ARGS__);         \
  if (!replace && !directed && disjoint && return_edge_id)                \
    return sample<false, false, true, true, false>(__VA_ARGS__);          \
  if (!replace && !directed && disjoint && !return_edge_id)               \
    return sample<false, false, true, false, false>(__VA_ARGS__);         \
  if (!replace && !directed && !disjoint && return_edge_id)               \
    return sample<false, false, false, true, false>(__VA_ARGS__);         \
  if (!replace && !directed && !disjoint && !return_edge_id)              \
    return sample<false, false, false, false, false>(__VA_ARGS__);

#define DISPATCH_DIST_SAMPLE(replace, directed, disjoint, ...)  \
  if (replace && directed && disjoint)                          \
    return sample<true, true, true, true, true>(__VA_ARGS__);   \
  if (replace && directed && !disjoint)                         \
    return sample<true, true, false, true, true>(__VA_ARGS__);  \
  if (replace && !directed && disjoint)                         \
    return sample<true, false, true, true, true>(__VA_ARGS__);  \
  if (replace && !directed && !disjoint)                        \
    return sample<true, false, false, true, true>(__VA_ARGS__); \
  if (!replace && directed && disjoint)                         \
    return sample<false, true, true, true, true>(__VA_ARGS__);  \
  if (!replace && directed && !disjoint)                        \
    return sample<false, true, false, true, true>(__VA_ARGS__); \
  if (!replace && !directed && disjoint)                        \
    return sample<false, false, true, true, true>(__VA_ARGS__); \
  if (!replace && !directed && !disjoint)                       \
    return sample<false, false, false, true, true>(__VA_ARGS__);

}  // namespace

std::tuple<at::Tensor,
           at::Tensor,
           at::Tensor,
           c10::optional<at::Tensor>,
           std::vector<int64_t>,
           std::vector<int64_t>>
neighbor_sample_kernel(const at::Tensor& rowptr,
                       const at::Tensor& col,
                       const at::Tensor& seed,
                       const std::vector<int64_t>& num_neighbors,
                       const c10::optional<at::Tensor>& time,
                       const c10::optional<at::Tensor>& seed_time,
                       const c10::optional<at::Tensor>& edge_weight,
                       bool csc,
                       bool replace,
                       bool directed,
                       bool disjoint,
                       std::string temporal_strategy,
                       bool return_edge_id,
                       bool old) {
  const auto out = [&] {
    DISPATCH_SAMPLE(replace, directed, disjoint, return_edge_id, rowptr, col,
                    seed, num_neighbors, time, seed_time, edge_weight, csc,
                    temporal_strategy, old);
  }();
  return std::make_tuple(std::get<0>(out), std::get<1>(out), std::get<2>(out),
                         std::get<3>(out), std::get<4>(out), std::get<5>(out));
}

std::tuple<c10::Dict<rel_type, at::Tensor>,
           c10::Dict<rel_type, at::Tensor>,
           c10::Dict<node_type, at::Tensor>,
           c10::optional<c10::Dict<rel_type, at::Tensor>>,
           c10::Dict<node_type, std::vector<int64_t>>,
           c10::Dict<rel_type, std::vector<int64_t>>>
hetero_neighbor_sample_kernel(
    const std::vector<node_type>& node_types,
    const std::vector<edge_type>& edge_types,
    const c10::Dict<rel_type, at::Tensor>& rowptr_dict,
    const c10::Dict<rel_type, at::Tensor>& col_dict,
    const c10::Dict<node_type, at::Tensor>& seed_dict,
    const c10::Dict<rel_type, std::vector<int64_t>>& num_neighbors_dict,
    const c10::optional<c10::Dict<node_type, at::Tensor>>& time_dict,
    const c10::optional<c10::Dict<node_type, at::Tensor>>& seed_time_dict,
    const c10::optional<c10::Dict<rel_type, at::Tensor>>& edge_weight_dict,
    bool csc,
    bool replace,
    bool directed,
    bool disjoint,
    std::string temporal_strategy,
    bool return_edge_id) {
  DISPATCH_SAMPLE(replace, directed, disjoint, return_edge_id, node_types,
                  edge_types, rowptr_dict, col_dict, seed_dict,
                  num_neighbors_dict, time_dict, seed_time_dict,
                  edge_weight_dict, csc, temporal_strategy);
}

std::tuple<at::Tensor, at::Tensor, std::vector<int64_t>>
dist_neighbor_sample_kernel(const at::Tensor& rowptr,
                            const at::Tensor& col,
                            const at::Tensor& seed,
                            const int64_t num_neighbors,
                            const c10::optional<at::Tensor>& time,
                            const c10::optional<at::Tensor>& seed_time,
                            const c10::optional<at::Tensor>& edge_weight,
                            bool csc,
                            bool replace,
                            bool directed,
                            bool disjoint,
                            std::string temporal_strategy,
                            bool old) {
  const auto out = [&] {
    DISPATCH_DIST_SAMPLE(replace, directed, disjoint, rowptr, col, seed,
                         {num_neighbors}, time, seed_time, edge_weight, csc,
                         temporal_strategy, old);
  }();
  return std::make_tuple(std::get<2>(out), std::get<3>(out).value(),
                         std::get<6>(out));
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

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::dist_neighbor_sample"),
         TORCH_FN(dist_neighbor_sample_kernel));
}

}  // namespace sampler
}  // namespace pyg
