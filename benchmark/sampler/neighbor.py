import argparse
import ast
import time
import csv

# import dgl
import torch
import torch_sparse  # noqa
from tqdm import tqdm

import pyg_lib
from pyg_lib.testing import withDataset, withSeed

from torch_geometric.data import Data
from torch_geometric.sampler.utils import to_csc

argparser = argparse.ArgumentParser()
argparser.add_argument('--batch-sizes', nargs='+', type=int, default=[
    # 3,
    512,
    1024,
    2048,
    4096,
    8192,
])
argparser.add_argument('--num_neighbors', type=ast.literal_eval, default=[
    # [2, 2],
    [-1],
    [15, 10, 5],
    [20, 15, 10],
])

argparser.add_argument('--replace', action='store_true')
argparser.add_argument('--directed', action='store_true')
argparser.add_argument('--shuffle', action='store_true')
argparser.add_argument('--write-csv', action='store_true')
argparser.add_argument('--pyg-lib-only', action='store_true')
args = argparser.parse_args()

# edge_index = torch.tensor([[0, 1, 0, 4, 1, 2, 2, 3, 2, 5, 3, 4, 3, 5, 6, 3, 6, 4, 6, 5, 6, 7, 7, 8, 8, 5],
#                            [1, 0, 4, 0, 2, 1, 3, 2, 5, 2, 4, 3, 5, 3, 3, 6, 4, 6, 5, 6, 7, 6, 8, 7, 5, 8]], dtype=torch.long)
# x = torch.tensor([[-1], [0], [1], [2], [3], [4], [5], [6], [7]], dtype=torch.float)

# data = Data(x=x, edge_index=edge_index, y=2)

# num_nodes = 9

@withSeed
@withDataset('DIMACS10', 'citationCiteseer')
def test_neighbor(dataset, **kwargs):
    (rowptr, col), num_nodes = dataset, dataset[0].size(0) - 1
    # dgl_graph = dgl.graph(('csc', (rowptr, col, torch.arange(col.size(0)))))

    # out = to_csc(data, device='cpu', share_memory=False,
    #                      is_sorted=False, src_node_time=None)
    # rowptr, col, _ = out

    if args.shuffle:
        node_perm = torch.randperm(num_nodes)
    else:
        node_perm = torch.arange(num_nodes)
    
    if args.write_csv:
        f = open('./neighbor_benchmark_parallel.csv', 'a', newline='')
        writer = csv.writer(f)
        writer.writerow(('num_neighbors', 'batch_size', 'pyg-lib', f'directed = {args.directed}'))


    for num_neighbors in args.num_neighbors:
        for batch_size in args.batch_sizes:
            print(f'batch_size={batch_size}, num_neighbors={num_neighbors}):')
            t = time.perf_counter()
            for seed in tqdm(node_perm.split(batch_size)):
                pyg_lib.sampler.neighbor_sample(
                    rowptr,
                    col,
                    seed,
                    num_neighbors,
                    replace=args.replace,
                    directed=args.directed,
                    disjoint=True,
                    return_edge_id=True,
                )
            pyg_lib_duration = time.perf_counter() - t
            
            # t = time.perf_counter()
            # for seed in tqdm(node_perm.split(batch_size)):
            #     torch.ops.torch_sparse.neighbor_sample(
            #         rowptr,
            #         col,
            #         seed,
            #         num_neighbors,
            #         args.replace,
            #         args.directed,
            #     )
            # torch_sparse_duration = time.perf_counter() - t

            # dgl_sampler = dgl.dataloading.NeighborSampler(
            #     num_neighbors,
            #     replace=args.replace,
            # )
            # dgl_loader = dgl.dataloading.DataLoader(
            #     dgl_graph,
            #     node_perm,
            #     dgl_sampler,
            #     batch_size=batch_size,
            # )
            # t = time.perf_counter()
            # for _ in tqdm(dgl_loader):
            #     pass
            # dgl_duration = time.perf_counter() - t

            pyg_lib_duration = round(pyg_lib_duration, 3)
            print(f'     pyg-lib={pyg_lib_duration} seconds')
            # print(f'torch-sparse={torch_sparse_duration:.3f} seconds')
            # print(f'         dgl={dgl_duration:.3f} seconds')
            print()
            if args.pyg_lib_only:
                pyg_lib_duration_coma = str(pyg_lib_duration).replace('.', ',')
                writer.writerow((num_neighbors, batch_size, pyg_lib_duration_coma))
    f.close()
if __name__ == '__main__':
    test_neighbor()
