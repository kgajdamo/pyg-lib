from typing import List, Tuple, Optional, Dict

import torch
from torch import Tensor

NodeType = str
RelType = str
EdgeType = Tuple[str, str, str]


def neighbor_sample(
    rowptr: Tensor,
    col: Tensor,
    seed: Tensor,
    num_neighbors: List[int],
    time: Optional[Tensor] = None,
    seed_time: Optional[Tensor] = None,
    batch: Optional[Tensor] = None,
    csc: bool = False,
    replace: bool = False,
    directed: bool = True,
    disjoint: bool = False,
    temporal_strategy: str = 'uniform',
    return_edge_id: bool = True,
    distributed: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], List[int], List[int],
           List[int]]:
    r"""Recursively samples neighbors from all node indices in :obj:`seed`
    in the graph given by :obj:`(rowptr, col)`.

    .. note::

        For temporal sampling, the :obj:`col` vector needs to be sorted
        according to :obj:`time` within individual neighborhoods since we use
        binary search to find neighbors that fulfill temporal constraints.

    Args:
        rowptr (torch.Tensor): Compressed source node indices.
        col (torch.Tensor): Target node indices.
        seed (torch.Tensor): The seed node indices.
        num_neighbors (List[int]): The number of neighbors to sample for each
            node in each iteration. If an entry is set to :obj:`-1`, all
            neighbors will be included.
        time (torch.Tensor, optional): Timestamps for the nodes in the graph.
            If set, temporal sampling will be used such that neighbors are
            guaranteed to fulfill temporal constraints, *i.e.* neighbors have
            an earlier or equal timestamp than the seed node.
            If used, the :obj:`col` vector needs to be sorted according to time
            within individual neighborhoods. Requires :obj:`disjoint=True`.
            (default: :obj:`None`)
        seed_time (torch.Tensor, optional): Optional values to override the
            timestamp for seed nodes. If not set, will use timestamps in
            :obj:`time` as default for seed nodes. (default: :obj:`None`)
        batch (torch.Tensor, optional): Optional values to specify the
            initial subgraph indices for seed nodes. If not set, will use
            incremental values starting from 0. (default: :obj:`None`)
        csc (bool, optional): If set to :obj:`True`, assumes that the graph is
            given in CSC format :obj:`(colptr, row)`. (default: :obj:`False`)
        replace (bool, optional): If set to :obj:`True`, will sample with
            replacement. (default: :obj:`False`)
        directed (bool, optional): If set to :obj:`False`, will include all
            edges between all sampled nodes. (default: :obj:`True`)
        disjoint (bool, optional): If set to :obj:`True` , will create disjoint
            subgraphs for every seed node. (default: :obj:`False`)
        temporal_strategy (string, optional): The sampling strategy when using
            temporal sampling (:obj:`"uniform"`, :obj:`"last"`).
            (default: :obj:`"uniform"`)
        return_edge_id (bool, optional): If set to :obj:`False`, will not
            return the indices of edges of the original graph.
            (default: :obj: `True`)
        distributed (bool, optional): If set to :obj:`True`, will sample nodes
            with duplicates, save information about the number of sampled
            neighbors per node and will not return rows and cols.
            This argument was added for the purpose of a distributed training.

    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor],
        List[int], List[int], List[int]):
        Row indices, col indices of the returned subtree/subgraph, as well as
        original node indices for all nodes sampled.
        In addition, may return the indices of edges of the original graph.
        Lastly, returns information about the sampled amount of nodes and edges
        per hop and if `distributed` will return cummulative sum of the sampled
        neighbors per node.
    """
    return torch.ops.pyg.neighbor_sample(rowptr, col, seed, num_neighbors,
                                         time, seed_time, batch, csc, replace,
                                         directed, disjoint, temporal_strategy,
                                         return_edge_id, distributed)


def hetero_neighbor_sample(
    rowptr_dict: Dict[EdgeType, Tensor],
    col_dict: Dict[EdgeType, Tensor],
    seed_dict: Dict[NodeType, Tensor],
    num_neighbors_dict: Dict[EdgeType, List[int]],
    time_dict: Optional[Dict[NodeType, Tensor]] = None,
    seed_time_dict: Optional[Dict[NodeType, Tensor]] = None,
    csc: bool = False,
    replace: bool = False,
    directed: bool = True,
    disjoint: bool = False,
    temporal_strategy: str = 'uniform',
    return_edge_id: bool = True,
    distributed: bool = False,
) -> Tuple[Dict[EdgeType, Tensor], Dict[EdgeType, Tensor], Dict[
        NodeType, Tensor], Optional[Dict[EdgeType, Tensor]], Dict[
            NodeType, List[int]], Dict[EdgeType, List[int]]]:
    r"""Recursively samples neighbors from all node indices in :obj:`seed_dict`
    in the heterogeneous graph given by :obj:`(rowptr_dict, col_dict)`.

    .. note ::
        Similar to :meth:`neighbor_sample`, but expects a dictionary of node
        types (:obj:`str`) and  edge types (:obj:`Tuple[str, str, str]`) for
        each non-boolean argument.

    Args:
        kwargs: Arguments of :meth:`neighbor_sample`.
    """
    src_node_types = {k[0] for k in rowptr_dict.keys()}
    dst_node_types = {k[-1] for k in rowptr_dict.keys()}
    node_types = list(src_node_types | dst_node_types)
    edge_types = list(rowptr_dict.keys())

    TO_REL_TYPE = {key: '__'.join(key) for key in edge_types}
    TO_EDGE_TYPE = {'__'.join(key): key for key in edge_types}

    rowptr_dict = {TO_REL_TYPE[k]: v for k, v in rowptr_dict.items()}
    col_dict = {TO_REL_TYPE[k]: v for k, v in col_dict.items()}
    num_neighbors_dict = {
        TO_REL_TYPE[k]: v
        for k, v in num_neighbors_dict.items()
    }

    out = torch.ops.pyg.hetero_neighbor_sample(
        node_types, edge_types, rowptr_dict, col_dict, seed_dict,
        num_neighbors_dict, time_dict, seed_time_dict, csc, replace, directed,
        disjoint, temporal_strategy, return_edge_id, distributed)

    (row_dict, col_dict, node_id_dict, edge_id_dict, num_nodes_per_hop_dict,
     num_edges_per_hop_dict) = out

    row_dict = {TO_EDGE_TYPE[k]: v for k, v in row_dict.items()}
    col_dict = {TO_EDGE_TYPE[k]: v for k, v in col_dict.items()}

    if edge_id_dict is not None:
        edge_id_dict = {TO_EDGE_TYPE[k]: v for k, v in edge_id_dict.items()}

    num_edges_per_hop_dict = {
        TO_EDGE_TYPE[k]: v
        for k, v in num_edges_per_hop_dict.items()
    }

    return (row_dict, col_dict, node_id_dict, edge_id_dict,
            num_nodes_per_hop_dict, num_edges_per_hop_dict)


def subgraph(
    rowptr: Tensor,
    col: Tensor,
    nodes: Tensor,
    return_edge_id: bool = True,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    r"""Returns the induced subgraph of the graph given by
    :obj:`(rowptr, col)`, containing only the nodes in :obj:`nodes`.

    Args:
        rowptr (torch.Tensor): Compressed source node indices.
        col (torch.Tensor): Target node indices.
        nodes (torch.Tensor): Node indices of the induced subgraph.
        return_edge_id (bool, optional): If set to :obj:`False`, will not
            return the indices of edges of the original graph contained in the
            induced subgraph. (default: :obj:`True`)

    Returns:
        (torch.Tensor, torch.Tensor, Optional[torch.Tensor]): Compressed source
        node indices and target node indices of the induced subgraph.
        In addition, may return the indices of edges of the original graph.
    """
    return torch.ops.pyg.subgraph(rowptr, col, nodes, return_edge_id)


def random_walk(rowptr: Tensor, col: Tensor, seed: Tensor, walk_length: int,
                p: float = 1.0, q: float = 1.0) -> Tensor:
    r"""Samples random walks of length :obj:`walk_length` from all node
    indices in :obj:`seed` in the graph given by :obj:`(rowptr, col)`, as
    described in the `"node2vec: Scalable Feature Learning for Networks"
    <https://arxiv.org/abs/1607.00653>`_ paper.

    Args:
        rowptr (torch.Tensor): Compressed source node indices.
        col (torch.Tensor): Target node indices.
        seed (torch.Tensor): Seed node indices from where random walks start.
        walk_length (int): The walk length of a random walk.
        p (float, optional): Likelihood of immediately revisiting a node in the
            walk. (default: :obj:`1.0`)
        q (float, optional): Control parameter to interpolate between
            breadth-first strategy and depth-first strategy.
            (default: :obj:`1.0`)

    Returns:
        torch.Tensor: A tensor of shape :obj:`[seed.size(0), walk_length + 1]`,
        holding the nodes indices of each walk for each seed node.
    """
    return torch.ops.pyg.random_walk(rowptr, col, seed, walk_length, p, q)


def relabel_neighborhood(
    seed: Tensor,
    sampled_nodes_with_dupl: Tensor,
    sampled_nbrs_per_node: List[int],
    num_nodes: int,
    batch: Optional[Tensor] = None,
    csc: bool = False,
    disjoint: bool = False,
) -> Tuple[Tensor, Tensor]:
    r"""Relabel global indices of the :obj:`sampled_nodes_with_dupl` to the
        local subtree/subgraph indices.

    .. note::

        For :obj:`disjoint`, the :obj:`batch` needs to be specified
        and each node from :obj:`sampled_nodes_with_dupl` must be assigned
        to a subgraph.

    Args:
        seed (torch.Tensor): The seed node indices.
        sampled_nodes_with_dupl (torch.Tensor): Sampled nodes with duplicates.
            Should not include seed nodes.
        sampled_nbrs_per_node (List[int]): The number of neighbors sampled by
            each node from :obj:`sampled_nodes_with_dupl`.
        num_nodes (int): Number of all nodes in a graph.
        batch (torch.Tensor, optional): Stores information about which subgraph
            the node from :obj:`sampled_nodes_with_dupl` belongs to.
            Must be specified when :obj:`disjoint`. (default: :obj:`None`)
        csc (bool, optional): If set to :obj:`True`, assumes that the graph is
            given in CSC format :obj:`(colptr, row)`. (default: :obj:`False`)
        disjoint (bool, optional): If set to :obj:`True` , will create disjoint
            subgraphs for every seed node. (default: :obj:`False`)

    Returns:
        (torch.Tensor, torch.Tensor):
        Row indices, col indices of the returned subtree/subgraph.
    """
    return torch.ops.pyg.relabel_neighborhood(seed, sampled_nodes_with_dupl,
                                              sampled_nbrs_per_node, num_nodes,
                                              batch, csc, disjoint)


def hetero_relabel_neighborhood(
    edge_types: List[EdgeType], seed_dict: Dict[NodeType, Tensor],
    sampled_nodes_with_dupl_dict: Dict[NodeType, Tensor],
    sampled_nbrs_per_node_dict: Dict[NodeType,
                                     List[int]], num_nodes_dict: Dict[NodeType,
                                                                      int],
    batch_dict: Optional[Dict[NodeType, Tensor]] = None, csc: bool = False,
    disjoint: bool = False
) -> Tuple[Dict[EdgeType, Tensor], Dict[EdgeType, Tensor]]:
    r"""Relabel global indices of the :obj:`sampled_nodes_with_dupl` to the
        local subtree/subgraph indices in the heterogeneous graph.

    .. note ::
        Similar to :meth:`relabel_neighborhood`, but expects a dictionary of
        node types (:obj:`str`) and edge types (:obj:`Tuple[str, str, str]`)
        for each non-boolean argument.

    Args:
        kwargs: Arguments of :meth:`relabel_neighborhood`.
    """

    src_node_types = {k[0] for k in sampled_nodes_with_dupl_dict.keys()}
    dst_node_types = {k[-1] for k in sampled_nodes_with_dupl_dict.keys()}
    node_types = list(src_node_types | dst_node_types)

    return torch.ops.pyg.hetero_relabel_neighborhood(
        node_types, edge_types, seed_dict, sampled_nodes_with_dupl_dict,
        sampled_nbrs_per_node_dict, num_nodes_dict, batch_dict, csc, disjoint)


__all__ = [
    'neighbor_sample',
    'hetero_neighbor_sample',
    'subgraph',
    'random_walk',
    'relabel_neighborhood',
    'hetero_relabel_neighborhood',
]
