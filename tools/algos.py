# Contents of package:
# Classes and functions related to ....
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import copy
import numpy as np
import networkx as nx
from collections import deque
import matplotlib.pyplot as plt
from dataclasses import dataclass
from .asc import AcousticScenario
from collections import defaultdict
from .base import Parameters, randmat

@dataclass
class Run:
    cfg: Parameters

    def launch(self):
        # Generate scenario
        c = self.cfg
        asc = AcousticScenario(cfg=c)
        Ryy, Rss, s, n, v = asc.setup()
        y = s + n + v  # Observed signal (centralized)
        d = np.array([s[c.Mk * k:c.Mk * k + c.D, ...] for k in range(c.K)])  # target signals
        
        # Generate tree
        if c.graphDiameter is not None:
            graph = generate_tree_with_diameter(c.K, c.graphDiameter)
        else:
            graph = nx.complete_graph(c.K)
            for (u, v) in graph.edges():
                graph.edges[u, v]['weight'] = np.random.random()
            graph = nx.minimum_spanning_tree(graph)

        if c.domain == 'time':
            metrics = self.launch_single_line(Ryy, Rss, y, d, asc, graph)
        elif c.domain == 'wola':
            print(f"Processing {c.nPosFreqs} frequency lines vectorized...")
            metrics = self.launch_vectorized(Ryy, Rss, y, d, asc, graph)

        # Post-process results
        self.plot_metrics(metrics)

    def plot_metrics(self, metrics):
        c = self.cfg
        fig, axes = plt.subplots(1, len(c.metricsToCompute))
        for ii, m in enumerate(c.metricsToCompute):
            ax = axes[ii] if len(c.metricsToCompute) > 1 else axes
            if any('danse' in alg for alg in c.algos):
                ax.set_yscale('log')
                for ii, alg in enumerate(metrics[m].keys()):
                    if m == 'msew' and alg in ['centralized', 'local','unprocessed']:
                        continue
                    if 'danse' in alg:
                        if metrics[m][alg] is not None:
                            if c.domain == 'wola':
                                data = np.mean(metrics[m][alg], axis=(0, 1))
                            elif c.domain == 'time':
                                data = np.mean(metrics[m][alg], axis=0)
                            ax.plot(data, label=alg)
                        else:
                            print(f"No {m} data for {alg}, skipping.")
                    else:
                        # Non-iterative algorithms
                        ax.axhline(y=np.mean(metrics[m][alg]), linestyle='--', label=alg, color=f'C{ii}')
                ax.set_xlim(0, c.maxDANSEiter)
            else:
                pass  # TODO
            ax.legend()
            ax.set_title(m)
        fig.suptitle(f'{c.observability}, {c.scmEstimation}')
        fig.tight_layout()
        plt.show()
        fig.set_size_inches(8.5, 3.5)

    def launch_single_line(self, Ryy, Rss, y, d, asc: AcousticScenario, G):
        """Launch algorithms for a single frequency line (or time-domain processing)."""
        c = self.cfg
        W_netWide = dict([(alg, [
            None for _ in range(c.K)
        ]) for alg in c.algos])  # Initialize node-speciifc filters dictionary
        metrics = dict([(metric, dict([
            (alg, np.zeros(c.K)) for alg in c.algos + ['unprocessed']
        ])) for metric in c.metricsToCompute])
        metrics['msed']['unprocessed'] = [
            np.mean(np.abs(d[k, ...] - y[c.Mk * k:c.Mk * k + c.D, ...]) ** 2)
            for k in range(c.K)
        ]
        for alg in c.algos:
            if alg == "centralized":
                Wcentr = np.linalg.inv(Ryy) @ Rss
                W_netWide[alg] = [
                    Wcentr[:, c.Mk * k:c.Mk * k + c.D] for k in range(c.K)
                ]
                if 'msed' in c.metricsToCompute:
                    for k in range(c.K):
                        dhatk = W_netWide[alg][k].conj().T @ y
                        metrics['msed'][alg][k] = np.mean(np.abs(d[k, ...] - dhatk) ** 2)
            elif alg == "local":
                for k in range(c.K):
                    Rykyk = Ryy[c.Mk * k:c.Mk * (k + 1), c.Mk * k:c.Mk * (k + 1)]
                    Rsksk = Rss[c.Mk * k:c.Mk * (k + 1), c.Mk * k:c.Mk * k + c.D]
                    W_netWide[alg][k] = np.linalg.inv(Rykyk) @ Rsksk
                    if 'msed' in c.metricsToCompute:
                        dhatk = W_netWide[alg][k].conj().T @ y[c.Mk * k:c.Mk * (k + 1), :]
                        metrics['msed'][alg][k] = np.mean(np.abs(d[k, ...] - dhatk) ** 2)
            elif alg == "dmwf":
                # Neighbor-specific fusion matrices
                Pk = [None for _ in range(c.K)]
                for q in range(c.K):
                    Ryqyq = Ryy[c.Mk * q:c.Mk * (q + 1), c.Mk * q:c.Mk * (q + 1)]
                    Rgqgqu = np.zeros((c.Mk, asc.oQq[q]), dtype=complex)
                    for p in range(c.K):
                        if p == q:
                            continue
                        Eqps = np.zeros((c.Mk, asc.oQq[q]), dtype=complex)
                        Eqps[:asc.Qkq[q, p], :asc.Qkq[q, p]] = np.eye(asc.Qkq[q, p])
                        Eqps[asc.Qkq[q, p]:, asc.Qkq[q, p]:] = np.ones((c.Mk- asc.Qkq[q, p], asc.oQq[q] - asc.Qkq[q, p]))
                        Rgqgqu += Ryy[c.Mk * q:c.Mk * (q + 1), c.Mk * p:c.Mk * (p + 1)] @ Eqps
                    Pk[q] = np.linalg.inv(Ryqyq) @ Rgqgqu
                # Estimation filters
                for k in range(c.K):
                    # ty = C^H.y
                    QkqNeighs = np.delete(asc.oQq, k)  # Remove k
                    Ck = np.zeros((c.M, c.Mk + int(np.sum(QkqNeighs))), dtype=complex)
                    Ck[c.Mk * k:c.Mk * (k + 1), :c.Mk] = np.eye(c.Mk)
                    idxNei = 0
                    for q in range(c.K):
                        if q != k:
                            idxBeg = c.Mk + int(np.sum(QkqNeighs[:idxNei]))
                            idxEnd = idxBeg + asc.oQq[q]
                            Ck[c.Mk * q:c.Mk * (q + 1), idxBeg:idxEnd] = Pk[q]
                            idxNei += 1
                    # Compute the filters
                    tRyy = Ck.conj().T @ Ryy @ Ck
                    tRss = Ck.conj().T @ Rss @ Ck
                    tW = np.linalg.inv(tRyy) @ tRss[:, :c.D]
                    W_netWide[alg][k] = Ck @ tW

                    # Compute metrics
                    if 'msew' in c.metricsToCompute:
                        metrics['msew'][alg][k] = np.mean(
                            np.abs(W_netWide[alg][k] - W_netWide['centralized'][k]) ** 2
                        )
                    if 'msed' in c.metricsToCompute:
                        dhatk = W_netWide[alg][k].conj().T @ y
                        metrics['msed'][alg][k] = np.mean(np.abs(d[k, ...] - dhatk) ** 2)

            elif alg == "tidmwf":
                for k in range(c.K):
                    upstreamNodes, upstreamNeighs = get_upstream_nodes(G, k)
                    downstreamNodes, downstreamNeighs = get_downstream_nodes(G, k)
                    Cqk = [None for _ in range(c.K)]
                    for q in range(c.K):
                        dim = c.Mk + c.Q * len(upstreamNeighs[q])
                        Cqk[q] = np.zeros((c.M, dim), dtype=complex)
                        Cqk[q][c.Mk * q:c.Mk * (q + 1), :c.Mk] = np.eye(c.Mk)
                    # Compute fusion matrices
                    Pk = [None for _ in range(c.K)]
                    for q in flatten_list(tree_levels(G, k)):
                        for ii, n in enumerate(upstreamNeighs[q]):
                            idxBeg = c.Mk + ii * c.Q
                            idxEnd = idxBeg + c.Q
                            Cqk[q][:, idxBeg:idxEnd] = Cqk[n] @ Pk[n]
                        if q != k:
                            # Compute Pk
                            Rhyqhyq = Cqk[q].conj().T @ Ryy @ Cqk[q]
                            hEq = np.zeros((c.M, c.Q))
                            hEq[
                                c.Mk * downstreamNeighs[q]:\
                                c.Mk * downstreamNeighs[q] + c.Q, :
                            ] = np.eye(c.Q)
                            Rhyqyktq = Cqk[q].conj().T @ Ryy @ hEq
                            Pk[q] = np.linalg.inv(Rhyqhyq) @ Rhyqyktq
                    # Compute estimation filter
                    tRyy = Cqk[k].conj().T @ Ryy @ Cqk[k]
                    tRss = Cqk[k].conj().T @ Rss @ Cqk[k]
                    W_netWide[alg][k] = Cqk[k] @ np.linalg.inv(tRyy) @ tRss[:, :c.D]
                    
                    # Compute metrics
                    if 'msew' in c.metricsToCompute:
                        metrics['msew'][alg][k] = np.mean(
                            np.abs(W_netWide[alg][k] - W_netWide['centralized'][k]) ** 2
                        )
                    if 'msed' in c.metricsToCompute:
                        dhatk = W_netWide[alg][k].conj().T @ y
                        metrics['msed'][alg][k] = np.mean(np.abs(d[k, ...] - dhatk) ** 2)
            elif 'danse' in alg:
                # Adapt metrics dimension
                for m in c.metricsToCompute:
                    metrics[m][alg] = np.zeros((c.K, c.maxDANSEiter))
                # Initialize the fusion matrices
                Pk = [randmat((c.Mk, c.Qd)) for _ in range(c.K)]
                WkkPrev = [np.zeros((c.Mk, c.Qd), dtype=complex) for _ in range(c.K)]
                u = 0  # updating node index
                for i in range(c.maxDANSEiter):
                    for k in range(c.K):
                        if alg.startswith("tidanse"):
                            Ck = np.zeros((c.M, c.Mk + c.Qd), dtype=complex)
                            Ck[c.Mk * k:c.Mk * (k + 1), :c.Mk] = np.eye(c.Mk)
                            for q in range(c.K):
                                if q != k:
                                    Ck[c.Mk * q:c.Mk * (q + 1), c.Mk:] = Pk[q]
                        else:
                            Ck = np.zeros((c.M, c.Mk + c.Qd * (c.K - 1)), dtype=complex)
                            Ck[c.Mk * k:c.Mk * (k + 1), :c.Mk] = np.eye(c.Mk)
                            idxNei = 0
                            for q in range(c.K):
                                if q != k:
                                    idxBeg = c.Mk + idxNei * c.Qd
                                    idxEnd = idxBeg + c.Qd
                                    Ck[c.Mk * q:c.Mk * (q + 1), idxBeg:idxEnd] = Pk[q]
                                    idxNei += 1
                        # Compute the filters
                        tRyy = Ck.conj().T @ Ryy @ Ck
                        tRss = Ck.conj().T @ Rss @ Ck
                        tW = np.linalg.pinv(tRyy) @ tRss
                        if alg.startswith("rsdanse"):
                            alpha = 1 / np.log10(i + 10)
                            tW[:c.Mk, :c.Qd] = (1 - alpha) * WkkPrev[k] + alpha * tW[:c.Mk, :c.Qd]
                            WkkPrev[k] = tW[:c.Mk, :c.Qd]
                        if k == u or alg.startswith("rsdanse"):
                            if alg.startswith("tidanse"):
                                Pk[k] = tW[:c.Mk, :c.Qd] @ np.linalg.pinv(tW[c.Mk:, :c.Qd])
                            else:
                                Pk[k] = tW[:c.Mk, :c.Qd]
                        W_netWide[alg][k] = Ck @ tW[:, :c.D]

                        # Compute metrics
                        if 'msew' in c.metricsToCompute:
                            metrics['msew'][alg][k, i] = np.mean(
                                np.abs(W_netWide[alg][k] - W_netWide['centralized'][k]) ** 2
                            )
                        if 'msed' in c.metricsToCompute:
                            dhatk = W_netWide[alg][k].conj().T @ y
                            metrics['msed'][alg][k, i] = np.mean(np.abs(d[k, ...] - dhatk) ** 2)
                    u = (u + 1) % c.K  # Update the node index for next iteration
            else:
                raise ValueError(f"Unknown algorithm: {alg}")
            
        return metrics


def tree_levels(G: nx.Graph, root):
    # Compute shortest path distances from root
    dist = nx.single_source_shortest_path_length(G, root)
    
    # Get the tree structure (list of leaves to root, per level)
    tree_structure = [[] for _ in range(max(dist.values()) + 1)]
    for q in G.nodes:
        tree_structure[dist[q]].append(q)
    tree_structure.reverse()  # Reverse the order to have root at the end
    
    return tree_structure


def flatten_list(l):
    """Flatten a list of lists."""
    return [item for sublist in l for item in sublist]

    
def get_upstream_nodes(G: nx.Graph, root):
    """GPT"""
    # Compute shortest path distances from root
    dist = nx.single_source_shortest_path_length(G, root)
    
    upstreamNodes = [None for _ in range(len(G.nodes))]
    upstreamNeighbors = [None for _ in range(len(G.nodes))]
    for q in G.nodes:
        # Nodes "away" from root from the perspective of q:
        nodes_away = []

        # BFS from q, but only go to nodes with higher distance from root
        visited = set()
        stack = [q]
        
        while stack:
            node = stack.pop()
            visited.add(node)
            
            for neighbor in G.neighbors(node):
                if neighbor in visited:
                    continue
                if dist.get(neighbor, float('inf')) > dist[q]:
                    stack.append(neighbor)
                    nodes_away.append(neighbor)

        # Filter out q and root explicitly, if needed
        upstreamNodes[q] = np.sort([n for n in nodes_away if n != q and n != root])
        # Get the upstream neighbors of q
        upstreamNeighbors[q] = np.sort([n for n in G.neighbors(q) if n in upstreamNodes[q]])
    
    return upstreamNodes, upstreamNeighbors


def get_downstream_nodes(G: nx.Graph, root):
    nNodes = G.number_of_nodes()
    downstreamNeighbors = [None] * nNodes  # Initialize all entries to None
    downstreamNodes = [[] for _ in range(nNodes)]

    visited = set()
    queue = deque([root])
    visited.add(root)

    while queue:
        current = queue.popleft()
        for neighbor in G.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                downstreamNeighbors[neighbor] = current  # current is "toward root" for neighbor
                queue.append(neighbor)
    
    # Find downstream nodes for each node
    for k in G.nodes:
        currIdx = k
        while downstreamNeighbors[currIdx] is not None:
            downstreamNodes[k].append(downstreamNeighbors[currIdx])
            currIdx = downstreamNeighbors[currIdx]

    return downstreamNodes, downstreamNeighbors


def generate_tree_with_diameter(N, E):
    """
    Generates a tree with N nodes and exact diameter E.
    
    Parameters:
        N (int): Number of nodes (N > E).
        E (int): Desired diameter (E >= 1).
    
    Returns:
        T (networkx.Graph): Tree with specified diameter.
    """
    assert E >= 1, "Diameter must be at least 1."
    assert N >= E + 1, "Need at least E+1 nodes for diameter E."

    tree = nx.Graph()

    # Step 1: create a path of length E (E+1 nodes)
    path_nodes = list(range(E + 1))
    for i in range(E):
        tree.add_edge(path_nodes[i], path_nodes[i+1])

    next_node = E + 1

    # Step 2: attach remaining nodes to internal nodes on the path
    attachable_nodes = path_nodes[1:-1]  # avoid attaching to endpoints to preserve diameter

    i = 0
    while next_node < N:
        attach_to = attachable_nodes[i % len(attachable_nodes)]
        tree.add_edge(attach_to, next_node)
        next_node += 1
        i += 1

    return tree
