import argparse
import ast
import time
from collections import defaultdict
from datetime import datetime
from itertools import product

import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.sampler.utils import to_csc
from tqdm import tqdm

import pyg_lib
from pyg_lib.testing import withDataset, withSeed

argparser = argparse.ArgumentParser()
argparser.add_argument(
    '--batch-sizes',
    nargs='+',
    type=int,
    default=[
        # 3,
        512,
        1024,
        2048,
        4096,
        8192,
    ])
argparser.add_argument('--directed', action='store_true')
argparser.add_argument('--disjoint', action='store_true')
argparser.add_argument(
    '--num_neighbors',
    type=ast.literal_eval,
    default=[
        # [2, 2, 2],
        [-1],
        [15, 10, 5],
        [20, 15, 10],
    ])
argparser.add_argument('--replace', action='store_true')
argparser.add_argument('--shuffle', action='store_true')
argparser.add_argument('--temporal', action='store_true')
argparser.add_argument('--temporal-strategy', choices=['uniform', 'last'],
                       default='uniform')
argparser.add_argument('--write-csv', action='store_true')
argparser.add_argument('--libraries', nargs="*", type=str,
                       default=['pyg-lib', 'torch-sparse', 'dgl'])
args = argparser.parse_args()

edge_index = torch.tensor(
    [[0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 7],
     [1, 4, 0, 2, 3, 1, 3, 7, 1, 2, 6, 7, 0, 5, 4, 3, 7, 2, 3, 6]],
    dtype=torch.long)
x = torch.tensor([[-1], [0], [1], [0], [1], [-1], [-1], [1]],
                 dtype=torch.float)

dat = Data(x=x, edge_index=edge_index, y=2)

num_nodes = 8


@withSeed
@withDataset('DIMACS10', 'citationCiteseer')
def test_neighbor(dataset, **kwargs):
    if args.temporal and not args.disjoint:
        raise ValueError(
            "Temporal sampling needs to create disjoint subgraphs")

    # (rowptr, col), num_nodes = dataset, dataset[0].size(0) - 1
    out = to_csc(dat, device='cpu', share_memory=False, is_sorted=False,
                 src_node_time=None)
    rowptr, col, _ = out

    if 'dgl' in args.libraries:
        import dgl
        dgl_graph = dgl.graph(
            ('csc', (rowptr, col, torch.arange(col.size(0)))))

    if args.temporal:
        # generate random timestamps
        node_time, _ = torch.sort(torch.randint(0, 100000, (num_nodes, )))
    else:
        node_time = None

    if args.shuffle:
        node_perm = torch.randperm(num_nodes)
    else:
        node_perm = torch.arange(num_nodes)

    data = defaultdict(list)
    for num_neighbors, batch_size in product(args.num_neighbors,
                                             args.batch_sizes):

        print(f'batch_size={batch_size}, num_neighbors={num_neighbors}):')
        data['num_neighbors'].append(num_neighbors)
        data['batch-size'].append(batch_size)

        if 'pyg-lib' in args.libraries:
            t = time.perf_counter()
            for seed in tqdm(node_perm.split(batch_size)):
                out_sample = pyg_lib.sampler.neighbor_sample(
                    rowptr,
                    col,
                    seed,
                    num_neighbors,
                    time=node_time,
                    seed_time=None,
                    replace=args.replace,
                    directed=args.directed,
                    disjoint=args.disjoint,
                    temporal_strategy=args.temporal_strategy,
                    return_edge_id=True,
                )
                print(out_sample[0])
                print(out_sample[1])
            pyg_lib_duration = time.perf_counter() - t
            data['pyg-lib'].append(round(pyg_lib_duration, 3))
            print(f'     pyg-lib={pyg_lib_duration:.3f} seconds')

        if not args.disjoint:
            if 'torch-sparse' in args.libraries:
                import torch_sparse  # noqa
                t = time.perf_counter()
                for seed in tqdm(node_perm.split(batch_size)):
                    torch.ops.torch_sparse.neighbor_sample(
                        rowptr,
                        col,
                        seed,
                        num_neighbors,
                        args.replace,
                        args.directed,
                    )
                torch_sparse_duration = time.perf_counter() - t
                data['torch-sparse'].append(round(torch_sparse_duration, 3))
                print(f'torch-sparse={torch_sparse_duration:.3f} seconds')

            if 'dgl' in args.libraries:
                import dgl
                dgl_sampler = dgl.dataloading.NeighborSampler(
                    num_neighbors,
                    replace=args.replace,
                )
                dgl_loader = dgl.dataloading.DataLoader(
                    dgl_graph,
                    node_perm,
                    dgl_sampler,
                    batch_size=batch_size,
                )
                t = time.perf_counter()
                for _ in tqdm(dgl_loader):
                    pass
                dgl_duration = time.perf_counter() - t
                data['dgl'].append(round(dgl_duration, 3))
                print(f'         dgl={dgl_duration:.3f} seconds')
        print()

    if args.write_csv:
        df = pd.DataFrame(data)
        df.to_csv(f'neighbor{datetime.now()}.csv', index=False)


if __name__ == '__main__':
    test_neighbor()
