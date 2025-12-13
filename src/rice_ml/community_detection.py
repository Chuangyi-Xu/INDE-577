"""
Community Detection (Graph Clustering)

This module implements a simple, dependency-free community detection algorithm:
Girvan–Newman (edge betweenness based divisive clustering).

API style is scikit-learn-like:
    - fit(G)
    - predict(G=None)
    - fit_predict(G)

Input graph formats supported:
    1) adjacency dict: {node: iterable_of_neighbors}
    2) edge list: list[tuple[u, v]]
    3) adjacency matrix: numpy.ndarray (square, 0/1 or weights; treated as unweighted)

Notes:
- The implementation assumes an undirected graph.
- For large graphs, Girvan–Newman is expensive (O(n*m^2) worst-case). This is intended
  for educational use / small-to-medium graphs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple, Union
import math
import random

import numpy as np


Node = Hashable
AdjDict = Dict[Node, List[Node]]
Edge = Tuple[Node, Node]


def _as_undirected_edge(u: Node, v: Node) -> Tuple[Node, Node]:
    return (u, v) if u <= v else (v, u)  # relies on comparable nodes in many cases


def _normalize_edge(u: Node, v: Node) -> Tuple[Node, Node]:
    # For arbitrary hashables that may not be comparable, fall back to string ordering
    if u == v:
        return (u, v)
    try:
        return (u, v) if u < v else (v, u)  # type: ignore[operator]
    except Exception:
        su, sv = str(u), str(v)
        return (u, v) if su < sv else (v, u)


def _build_adj_from_edges(edges: Sequence[Edge]) -> AdjDict:
    adj: AdjDict = {}
    for u, v in edges:
        if u not in adj:
            adj[u] = []
        if v not in adj:
            adj[v] = []
        if v not in adj[u]:
            adj[u].append(v)
        if u not in adj[v]:
            adj[v].append(u)
    return adj


def _build_adj_from_matrix(A: np.ndarray) -> AdjDict:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Adjacency matrix must be square (n x n).")
    n = A.shape[0]
    adj: AdjDict = {i: [] for i in range(n)}
    # treat as unweighted: any nonzero means edge
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] != 0 or A[j, i] != 0:
                adj[i].append(j)
                adj[j].append(i)
    return adj


def _build_adj(G: Union[AdjDict, Sequence[Edge], np.ndarray]) -> AdjDict:
    if isinstance(G, dict):
        # copy & make sure lists
        adj: AdjDict = {k: list(v) for k, v in G.items()}
        # enforce undirected symmetry (best-effort)
        for u, nbrs in list(adj.items()):
            if u not in adj:
                adj[u] = []
            for v in nbrs:
                if v not in adj:
                    adj[v] = []
                if u not in adj[v]:
                    adj[v].append(u)
        return adj
    if isinstance(G, np.ndarray):
        return _build_adj_from_matrix(G)
    # assume edge list
    return _build_adj_from_edges(G)


def _connected_components(adj: AdjDict) -> List[List[Node]]:
    seen = set()
    comps: List[List[Node]] = []

    for start in adj.keys():
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp: List[Node] = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj.get(u, []):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(comp)

    return comps


def _edge_set(adj: AdjDict) -> List[Tuple[Node, Node]]:
    edges = []
    for u, nbrs in adj.items():
        for v in nbrs:
            e = _normalize_edge(u, v)
            if e[0] == e[1]:
                continue
            edges.append(e)
    # unique
    return list(set(edges))


def _brandes_edge_betweenness(adj: AdjDict) -> Dict[Tuple[Node, Node], float]:
    """
    Brandes algorithm for edge betweenness centrality in unweighted graphs.
    Returns dict edge -> betweenness score.
    """
    nodes = list(adj.keys())
    bet: Dict[Tuple[Node, Node], float] = {e: 0.0 for e in _edge_set(adj)}

    for s in nodes:
        # BFS tree
        stack: List[Node] = []
        pred: Dict[Node, List[Node]] = {v: [] for v in nodes}
        sigma: Dict[Node, float] = {v: 0.0 for v in nodes}
        dist: Dict[Node, int] = {v: -1 for v in nodes}

        sigma[s] = 1.0
        dist[s] = 0
        queue: List[Node] = [s]

        qi = 0
        while qi < len(queue):
            v = queue[qi]
            qi += 1
            stack.append(v)
            for w in adj.get(v, []):
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        # accumulation
        delta: Dict[Node, float] = {v: 0.0 for v in nodes}
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w] == 0:
                    continue
                c = (sigma[v] / sigma[w]) * (1.0 + delta[w])
                e = _normalize_edge(v, w)
                bet[e] = bet.get(e, 0.0) + c
                delta[v] += c

    # undirected graph => divide by 2
    for e in list(bet.keys()):
        bet[e] *= 0.5

    return bet


def _remove_edge(adj: AdjDict, u: Node, v: Node) -> None:
    if v in adj.get(u, []):
        adj[u].remove(v)
    if u in adj.get(v, []):
        adj[v].remove(u)


def _modularity(adj_original: AdjDict, communities: List[List[Node]]) -> float:
    """
    Modularity Q for undirected, unweighted graph.
    Q = (1/2m) sum_{ij} [A_ij - (k_i k_j)/(2m)] * 1[c_i=c_j]
    """
    # degrees + m
    deg: Dict[Node, int] = {u: len(adj_original.get(u, [])) for u in adj_original}
    m = sum(deg.values()) / 2.0
    if m == 0:
        return 0.0

    # build adjacency lookup for A_ij
    edge_lookup = set(_edge_set(adj_original))

    Q = 0.0
    two_m = 2.0 * m

    for comm in communities:
        comm_set = set(comm)
        # sum over i,j in community
        for i in comm_set:
            for j in comm_set:
                if i == j:
                    continue
                Aij = 1.0 if _normalize_edge(i, j) in edge_lookup else 0.0
                Q += (Aij - (deg[i] * deg[j]) / two_m)

    Q /= two_m
    return Q


@dataclass
class GirvanNewmanCommunityDetection:
    """
    Girvan–Newman community detection (divisive clustering).

    Parameters
    ----------
    n_communities : int, optional
        If provided, stop when number of connected components reaches this value.
    max_remove_steps : int
        Hard cap on how many edge-removal iterations to run.
    keep_best_by_modularity : bool
        If True, track the partition with highest modularity and return that.
        If False, return the partition at the stopping condition.
    random_state : int, optional
        Used only for tie-breaking when multiple edges share max betweenness.
    """

    n_communities: Optional[int] = None
    max_remove_steps: int = 200
    keep_best_by_modularity: bool = True
    random_state: Optional[int] = None

    # learned attributes
    communities_: Optional[List[List[Node]]] = None
    labels_: Optional[Dict[Node, int]] = None
    modularity_: Optional[float] = None
    nodes_: Optional[List[Node]] = None

    def fit(self, G: Union[AdjDict, Sequence[Edge], np.ndarray]) -> "GirvanNewmanCommunityDetection":
        rng = random.Random(self.random_state)

        adj0 = _build_adj(G)
        self.nodes_ = list(adj0.keys())

        # working copy for edge removals
        adj = {u: list(vs) for u, vs in adj0.items()}

        # initial partition
        comps = _connected_components(adj)
        best_comps = comps
        best_Q = _modularity(adj0, comps)

        steps = 0
        while steps < self.max_remove_steps:
            steps += 1

            # stopping criterion (if user requests a target number of communities)
            if self.n_communities is not None and len(comps) >= self.n_communities:
                if not self.keep_best_by_modularity:
                    best_comps = comps
                    best_Q = _modularity(adj0, comps)
                break

            edges = _edge_set(adj)
            if not edges:
                break

            bet = _brandes_edge_betweenness(adj)
            if not bet:
                break

            max_b = max(bet.values())
            candidates = [e for e, b in bet.items() if abs(b - max_b) < 1e-12]
            # tie-break
            e_rm = rng.choice(candidates)

            _remove_edge(adj, e_rm[0], e_rm[1])

            comps = _connected_components(adj)
            Q = _modularity(adj0, comps)

            if self.keep_best_by_modularity and Q > best_Q + 1e-12:
                best_Q = Q
                best_comps = comps

        # finalize
        self.communities_ = [sorted(c, key=lambda x: str(x)) for c in best_comps]
        self.modularity_ = float(best_Q)

        labels: Dict[Node, int] = {}
        for cid, comm in enumerate(self.communities_):
            for node in comm:
                labels[node] = cid
        self.labels_ = labels

        return self

    def predict(self, G: Optional[Union[AdjDict, Sequence[Edge], np.ndarray]] = None) -> np.ndarray:
        """
        Return community labels aligned to the node ordering.

        If G is None, uses the node ordering from fit().
        If G is provided, labels will be returned in the ordering of G's nodes.
        Nodes unseen during fit() will be assigned label -1.
        """
        if self.labels_ is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        if G is None:
            if self.nodes_ is None:
                raise RuntimeError("No stored node ordering. Fit the model first.")
            nodes = self.nodes_
        else:
            adj = _build_adj(G)
            nodes = list(adj.keys())

        y = np.array([self.labels_.get(n, -1) for n in nodes], dtype=int)
        return y

    def fit_predict(self, G: Union[AdjDict, Sequence[Edge], np.ndarray]) -> np.ndarray:
        return self.fit(G).predict(G)

    def get_communities(self) -> List[List[Node]]:
        if self.communities_ is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        return self.communities_
