#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include "parallel_hashmap/phmap.h"

#include "pyg_lib/csrc/sampler/cpu/dist_relabel_kernel.h"
#include "pyg_lib/csrc/sampler/cpu/mapper.h"
#include "pyg_lib/csrc/utils/cpu/convert.h"
#include "pyg_lib/csrc/utils/types.h"

namespace pyg {
namespace sampler {

namespace {

template <typename scalar_t>
std::tuple<at::Tensor, at::Tensor> get_sampled_edges(
    std::vector<scalar_t> sampled_rows,
    std::vector<scalar_t> sampled_cols,
    const bool csc = false) {
  const auto row = pyg::utils::from_vector(sampled_rows);
  const auto col = pyg::utils::from_vector(sampled_cols);

  if (!csc) {
    return std::make_tuple(row, col);
  } else {
    return std::make_tuple(col, row);
  }
}

template <bool disjoint>
std::tuple<at::Tensor, at::Tensor> relabel(
    const at::Tensor& seed,
    const at::Tensor& sampled_nodes_with_dupl,
    const std::vector<int64_t>& sampled_nbrs_per_node,
    const int64_t num_nodes,
    const c10::optional<at::Tensor>& batch,
    const bool csc) {
  if (disjoint) {
    TORCH_CHECK(batch.has_value(),
                "Batch needs to be specified to create disjoint subgraphs");
    TORCH_CHECK(batch.value().is_contiguous(), "Non-contiguous 'batch'");
    TORCH_CHECK(batch.value().numel() == sampled_nodes_with_dupl.numel(),
                "Each node must belong to a subgraph.'");
  }
  TORCH_CHECK(seed.is_contiguous(), "Non-contiguous 'seed'");
  TORCH_CHECK(sampled_nodes_with_dupl.is_contiguous(),
              "Non-contiguous 'sampled_nodes_with_dupl'");

  at::Tensor out_row, out_col;

  AT_DISPATCH_INTEGRAL_TYPES(
      seed.scalar_type(), "relabel_neighborhood_kernel", [&] {
        typedef std::pair<scalar_t, scalar_t> pair_scalar_t;
        typedef std::conditional_t<!disjoint, scalar_t, pair_scalar_t> node_t;

        const auto sampled_nodes_data =
            sampled_nodes_with_dupl.data_ptr<scalar_t>();
        const auto batch_data =
            !disjoint ? nullptr : batch.value().data_ptr<scalar_t>();

        std::vector<scalar_t> sampled_rows;
        std::vector<scalar_t> sampled_cols;
        auto mapper = Mapper<node_t, scalar_t>(num_nodes);

        const auto seed_data = seed.data_ptr<scalar_t>();
        if constexpr (!disjoint) {
          mapper.fill(seed);
        } else {
          for (size_t i = 0; i < seed.numel(); ++i) {
            mapper.insert({i, seed_data[i]});
          }
        }
        size_t begin = 0, end = 0;
        for (auto i = 0; i < sampled_nbrs_per_node.size(); i++) {
          end += sampled_nbrs_per_node[i];

          for (auto j = begin; j < end; j++) {
            std::pair<scalar_t, bool> res;
            if constexpr (!disjoint)
              res = mapper.insert(sampled_nodes_data[j]);
            else
              res = mapper.insert({batch_data[j], sampled_nodes_data[j]});
            sampled_rows.push_back(i);
            sampled_cols.push_back(res.first);
          }

          begin = end;
        }

        std::tie(out_row, out_col) =
            get_sampled_edges<scalar_t>(sampled_rows, sampled_cols, csc);
      });

  return std::make_tuple(out_row, out_col);
}

template <bool disjoint>
std::tuple<c10::Dict<rel_type, at::Tensor>, c10::Dict<rel_type, at::Tensor>>
relabel(
    const std::vector<node_type>& node_types,
    const std::vector<edge_type>& edge_types,
    const c10::Dict<node_type, at::Tensor>& seed_dict,
    const c10::Dict<node_type, at::Tensor>& sampled_nodes_with_dupl_dict,
    const c10::Dict<rel_type, std::vector<int64_t>>& sampled_nbrs_per_node_dict,
    const c10::Dict<node_type, int64_t> num_nodes_dict,
    const c10::optional<c10::Dict<node_type, at::Tensor>>& batch_dict,
    const bool csc) {
  if (disjoint) {
    TORCH_CHECK(batch_dict.has_value(),
                "Batch needs to be specified to create disjoint subgraphs");
    for (const auto& kv : batch_dict.value()) {
      const at::Tensor& batch = kv.value();
      const at::Tensor& sampled_nodes_with_dupl = kv.value();
      TORCH_CHECK(batch.is_contiguous(), "Non-contiguous 'batch'");
      TORCH_CHECK(batch.numel() == sampled_nodes_with_dupl.numel(),
                  "Each node must belong to a subgraph.'");
    }
  }
  for (const auto& kv : seed_dict) {
    const at::Tensor& seed = kv.value();
    TORCH_CHECK(seed.is_contiguous(), "Non-contiguous 'seed'");
  }
  for (const auto& kv : sampled_nodes_with_dupl_dict) {
    const at::Tensor& sampled_nodes_with_dupl = kv.value();
    TORCH_CHECK(sampled_nodes_with_dupl.is_contiguous(),
                "Non-contiguous 'sampled_nodes_with_dupl'");
  }

  c10::Dict<rel_type, at::Tensor> out_row_dict, out_col_dict;

  AT_DISPATCH_INTEGRAL_TYPES(
      seed_dict.begin()->value().scalar_type(),
      "hetero_relabel_neighborhood_kernel", [&] {
        typedef std::pair<scalar_t, scalar_t> pair_scalar_t;
        typedef std::conditional_t<!disjoint, scalar_t, pair_scalar_t> node_t;

        phmap::flat_hash_map<node_type, scalar_t*> sampled_nodes_data_dict;
        phmap::flat_hash_map<node_type, scalar_t*> batch_data_dict;
        phmap::flat_hash_map<edge_type, std::vector<scalar_t>>
            sampled_rows_dict;
        phmap::flat_hash_map<edge_type, std::vector<scalar_t>>
            sampled_cols_dict;

        phmap::flat_hash_map<node_type, Mapper<node_t, scalar_t>> mapper_dict;
        phmap::flat_hash_map<node_type, std::pair<size_t, size_t>> slice_dict;

        for (const auto& k : edge_types) {
          // Initialize empty vectors.
          sampled_rows_dict[k];
          sampled_cols_dict[k];
        }
        for (const auto& k : node_types) {
          sampled_nodes_data_dict.insert(
              {k, sampled_nodes_with_dupl_dict.at(k).data_ptr<scalar_t>()});
          const auto N = num_nodes_dict.at(k) > 0 ? num_nodes_dict.at(k) : 0;
          mapper_dict.insert({k, Mapper<node_t, scalar_t>(N)});
          slice_dict[k] = {0, 0};
          if constexpr (disjoint) {
            batch_data_dict.insert(
                {k, batch_dict.value().at(k).data_ptr<scalar_t>()});
          }
        }
        for (const auto& kv : seed_dict) {
          const at::Tensor& seed = kv.value();
          if constexpr (!disjoint) {
            mapper_dict.at(kv.key()).fill(seed);
          } else {
            auto& mapper = mapper_dict.at(kv.key());
            const auto seed_data = seed.data_ptr<scalar_t>();
            for (size_t i = 0; i < seed.numel(); ++i) {
              mapper.insert({i, seed_data[i]});
            }
          }
        }
        for (const auto& k : edge_types) {
          const auto src = !csc ? std::get<0>(k) : std::get<2>(k);
          const auto dst = !csc ? std::get<2>(k) : std::get<0>(k);
          for (auto i = 0;
               i < sampled_nbrs_per_node_dict.at(to_rel_type(k)).size(); i++) {
            auto& dst_mapper = mapper_dict.at(dst);
            auto& dst_sampled_nodes_data = sampled_nodes_data_dict.at(dst);
            slice_dict.at(dst).second +=
                sampled_nbrs_per_node_dict.at(to_rel_type(k))[i];
            size_t begin, end;
            std::tie(begin, end) = slice_dict.at(dst);

            for (auto j = begin; j < end; j++) {
              std::pair<scalar_t, bool> res;
              if constexpr (!disjoint) {
                res = dst_mapper.insert(dst_sampled_nodes_data[j]);
              } else {
                res = dst_mapper.insert(
                    {batch_data_dict.at(dst)[j], dst_sampled_nodes_data[j]});
              }
              sampled_rows_dict.at(k).push_back(i);
              sampled_cols_dict.at(k).push_back(res.first);
            }
            slice_dict.at(dst).first = end;
          }
        }

        for (const auto& k : edge_types) {
          const auto edges = get_sampled_edges<scalar_t>(
              sampled_rows_dict.at(k), sampled_cols_dict.at(k), csc);
          out_row_dict.insert(to_rel_type(k), std::get<0>(edges));
          out_col_dict.insert(to_rel_type(k), std::get<1>(edges));
        }
      });

  return std::make_tuple(out_row_dict, out_col_dict);
}

#define DISPATCH_RELABEL(disjoint, ...) \
  if (disjoint)                         \
    return relabel<true>(__VA_ARGS__);  \
  if (!disjoint)                        \
    return relabel<false>(__VA_ARGS__);

}  // namespace

std::tuple<at::Tensor, at::Tensor> relabel_neighborhood_kernel(
    const at::Tensor& seed,
    const at::Tensor& sampled_nodes_with_dupl,
    const std::vector<int64_t>& sampled_nbrs_per_node,
    const int64_t num_nodes,
    const c10::optional<at::Tensor>& batch,
    bool csc,
    bool disjoint) {
  DISPATCH_RELABEL(disjoint, seed, sampled_nodes_with_dupl,
                   sampled_nbrs_per_node, num_nodes, batch, csc);
}

std::tuple<c10::Dict<rel_type, at::Tensor>, c10::Dict<rel_type, at::Tensor>>
hetero_relabel_neighborhood_kernel(
    const std::vector<node_type>& node_types,
    const std::vector<edge_type>& edge_types,
    const c10::Dict<node_type, at::Tensor>& seed_dict,
    const c10::Dict<node_type, at::Tensor>& sampled_nodes_with_dupl_dict,
    const c10::Dict<rel_type, std::vector<int64_t>>& sampled_nbrs_per_node_dict,
    const c10::Dict<node_type, int64_t> num_nodes_dict,
    const c10::optional<c10::Dict<node_type, at::Tensor>>& batch_dict,
    bool csc,
    bool disjoint) {
  c10::Dict<rel_type, at::Tensor> out_row_dict, out_col_dict;
  DISPATCH_RELABEL(disjoint, node_types, edge_types, seed_dict,
                   sampled_nodes_with_dupl_dict, sampled_nbrs_per_node_dict,
                   num_nodes_dict, batch_dict, csc);
}

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::relabel_neighborhood"),
         TORCH_FN(relabel_neighborhood_kernel));
}

TORCH_LIBRARY_IMPL(pyg, BackendSelect, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::hetero_relabel_neighborhood"),
         TORCH_FN(hetero_relabel_neighborhood_kernel));
}

}  // namespace sampler
}  // namespace pyg
