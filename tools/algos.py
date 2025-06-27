# Contents of package:
# Classes and functions related to ....
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import copy
import numpy as np
import networkx as nx
import scipy.linalg as sla
from .base import Parameters
from collections import deque
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Run:
    cfg: Parameters
    obsMat: np.ndarray = None  # Observability matrix, initialized later
    oQq: np.ndarray = None  # Number of sources in common between nodes

    def setup(self):
        c = self.cfg

        Amat = randmat((c.M, c.Qd))
        Bmat = randmat((c.M, c.Qn))
        cAmat = [copy.deepcopy(Amat) for _ in range(c.K)]
        cBmat = [copy.deepcopy(Bmat) for _ in range(c.K)]
        slat = np.random.randn(c.Qd, c.N)
        nlat = np.random.randn(c.Qn, c.N)
        pows = np.mean(np.abs(slat) ** 2, axis=1)
        pown = np.mean(np.abs(nlat) ** 2, axis=1)
        # Compute steering matrices
        if c.observability == 'foss':
            self.oQq = np.full(c.K, c.Q)
            self.Qkq = np.full((c.K, c.K), c.Q)
        elif c.observability == 'poss':
            # Do not differentiate between global and local sources, 
            # randomly generate observability pattern
            self.obsMat = np.zeros((c.K, c.Q))
            def inadequate(om):
                return np.any(np.sum(om, axis=1) == 0) | np.any(np.sum(om, axis=0) == 0)
            # Criterion for adequacy: at least one desired source and one noise
            # source must be observed by each node, and each source must be
            # observed by at least one node.
            # while inadequate(self.obsMat[:, :c.Qd]) or inadequate(self.obsMat[:, c.Qd:]):
            while inadequate(self.obsMat):
                self.obsMat = np.random.randint(0, 2, (c.K, c.Q))
            if c.possDiffuse:
                # Make sure each noise source is only observed by one node at most
                for n in range(c.Qn):
                    idx = np.where(self.obsMat[:, c.Qd + n] == 1)[0]
                    if len(idx) > 1:
                        idx = np.random.choice(idx, 1, replace=False)
                        self.obsMat[:, c.Qd + n] = 0
                        self.obsMat[idx, c.Qd + n] = 1
            for k in range(c.K):
                for s in range(c.Qd):
                    if self.obsMat[k, s] == 0:
                        Amat[c.Mk * k:c.Mk * (k + 1), s] = 0
                for n in range(c.Qn):
                    if self.obsMat[k, c.Qd + n] == 0:
                        Bmat[c.Mk * k:c.Mk * (k + 1), n] = 0
            # Number of sources useful for fusion matrix computation for node q
            self.oQq = [0 for _ in range(c.K)]
            for k in range(c.K):
                for s in range(c.Q):
                    if self.obsMat[k, s] != 0 and np.sum(self.obsMat[:, s]) > 1:
                        # If node k does observes source s, and it is observed
                        # by at least one other node, then the number of sources
                        # in common is increased by one
                        self.oQq[k] += 1
            assert np.all(np.array(self.oQq) <= np.sum(self.obsMat, axis=1)), \
                "Number of sources in common exceeds number of sources observed by node."
            # Compute the number of sources in common between nodes k and q
            self.Qkq = np.zeros((c.K, c.K), dtype=int)
            for k in range(c.K):
                for q in range(c.K):
                    if k == q:
                        continue
                    self.Qkq[k, q] = np.sum(
                        self.obsMat[k, :] * self.obsMat[q, :]
                    )
            for k in range(c.K):
                # "Global" Amat and Bmat matrices
                # List of sources that are either not observed by node k, or
                # not observed by any other node
                idxUncorr_A, idxUncorr_B = [], []
                for s in range(c.Qd):
                    if self.obsMat[k, s] == 0 or np.sum(self.obsMat[:, s]) == 1:
                        idxUncorr_A.append(s)
                for n in range(c.Qn):
                    if self.obsMat[k, c.Qd + n] == 0 or np.sum(self.obsMat[:, c.Qd + n]) == 1:
                        idxUncorr_B.append(n)
                if len(idxUncorr_A) > 0:
                    cAmat[k][:, idxUncorr_A] = 0
                if len(idxUncorr_B) > 0:
                    cBmat[k][:, idxUncorr_B] = 0

        # Compute signals
        s = Amat @ slat
        n = Bmat @ nlat
        v = np.random.randn(c.M, c.N) * np.mean(pows) * c.selfNoiseFactor  # small self-noise
        
        # Compute the SCMs
        if c.scmEstimation == 'oracle':
            # For oracle SCM estimation, we assume perfect knowledge of the
            # source and noise steering matrices
            Gam_s = np.diag(pows)
            Gam_n = np.diag(pown)
            Rss = Amat @ Gam_s @ Amat.T
            Rnn = Bmat @ Gam_n @ Bmat.T
            Rvv = np.eye(c.M) * np.mean(pows) * c.selfNoiseFactor  # small self-noise
        elif c.scmEstimation == 'batch':
            # Batch SCM estimation based on actual signals
            Rss = s @ s.T / c.N
            Rnn = n @ n.T / c.N
            Rvv = v @ v.T / c.N
        
        # Complete signal SCM
        Ryy = Rss + Rnn + Rvv

        return Ryy, Rss, s, n, v

    def launch(self):
        # Generate scenario
        c = self.cfg
        Ryy, Rss, s, n, v = self.setup()
        y = s + n + v  # Observed signal (centralized)
        d = np.array([s[c.Mk * k:c.Mk * k + c.D, :] for k in range(c.K)])  # target signals
        
        # Generate tree
        if c.graphDiameter is not None:
            G = generate_tree_with_diameter(c.K, c.graphDiameter)
        else:
            G = nx.complete_graph(c.K)
            for (u, v) in G.edges():
                G.edges[u, v]['weight'] = np.random.random()
            G = nx.minimum_spanning_tree(G)

        W_netWide = dict([(alg, [
            None for _ in range(c.K)
        ]) for alg in c.algos])  # Initialize node-speciifc filters dictionary
        metrics = dict([(metric, dict([
            (alg, np.zeros(c.K)) for alg in c.algos + ['unprocessed']
        ])) for metric in c.metricsToCompute])
        metrics['msed']['unprocessed'] = [
            np.mean(np.abs(d[k, :] - y[c.Mk * k:c.Mk * k + c.D, :]) ** 2)
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
                        dhatk = W_netWide[alg][k].T @ y
                        metrics['msed'][alg][k] = np.mean(np.abs(d[k, :] - dhatk) ** 2)
            elif alg == "local":
                for k in range(c.K):
                    Rykyk = Ryy[c.Mk * k:c.Mk * (k + 1), c.Mk * k:c.Mk * (k + 1)]
                    Rsksk = Rss[c.Mk * k:c.Mk * (k + 1), c.Mk * k:c.Mk * k + c.D]
                    W_netWide[alg][k] = np.linalg.inv(Rykyk) @ Rsksk
                    if 'msed' in c.metricsToCompute:
                        dhatk = W_netWide[alg][k].T @ y[c.Mk * k:c.Mk * (k + 1), :]
                        metrics['msed'][alg][k] = np.mean(np.abs(d[k, :] - dhatk) ** 2)
            elif alg == "dmwf":
                # Neighbor-specific fusion matrices
                Pk = [None for _ in range(c.K)]
                for q in range(c.K):
                    Ryqyq = Ryy[c.Mk * q:c.Mk * (q + 1), c.Mk * q:c.Mk * (q + 1)]
                    Rgqgqu = np.zeros((c.Mk, self.oQq[q]))
                    for p in range(c.K):
                        if p == q:
                            continue
                        Eqps = np.zeros((c.Mk, self.oQq[q]))
                        Eqps[:self.Qkq[q, p], :self.Qkq[q, p]] = np.eye(self.Qkq[q, p])
                        # Eqps[self.Qkq[q, p]:, self.Qkq[q, p]:] = np.random.randn(c.Mk- self.Qkq[q, p], self.oQq[q] - self.Qkq[q, p])
                        Eqps[self.Qkq[q, p]:, self.Qkq[q, p]:] = np.ones((c.Mk- self.Qkq[q, p], self.oQq[q] - self.Qkq[q, p]))
                        Rgqgqu += Ryy[c.Mk * q:c.Mk * (q + 1), c.Mk * p:c.Mk * (p + 1)] @ Eqps
                    Pk[q] = np.linalg.inv(Ryqyq) @ Rgqgqu
                # Estimation filters
                for k in range(c.K):
                    # ty = C^H.y
                    QkqNeighs = np.delete(self.oQq, k)  # Remove k
                    Ck = np.zeros((c.M, c.Mk + int(np.sum(QkqNeighs))))
                    Ck[c.Mk * k:c.Mk * (k + 1), :c.Mk] = np.eye(c.Mk)
                    idxNei = 0
                    for q in range(c.K):
                        if q != k:
                            idxBeg = c.Mk + int(np.sum(QkqNeighs[:idxNei]))
                            idxEnd = idxBeg + self.oQq[q]
                            Ck[c.Mk * q:c.Mk * (q + 1), idxBeg:idxEnd] = Pk[q]
                            idxNei += 1
                    # Compute the filters
                    tRyy = Ck.T @ Ryy @ Ck
                    tRss = Ck.T @ Rss @ Ck
                    tW = np.linalg.inv(tRyy) @ tRss[:, :c.D]
                    W_netWide[alg][k] = Ck @ tW

                    # Compute metrics
                    if 'msew' in c.metricsToCompute:
                        metrics['msew'][alg][k] = np.mean(
                            np.abs(W_netWide[alg][k] - W_netWide['centralized'][k]) ** 2
                        )
                    if 'msed' in c.metricsToCompute:
                        dhatk = W_netWide[alg][k].T @ y
                        metrics['msed'][alg][k] = np.mean(np.abs(d[k, :] - dhatk) ** 2)
                        pass

            elif alg == "tidmwf":
                if c.observability == 'poss':
                    print("Warning: TI-dMWF is not implemented for partially overlapping subspaces.")
                    c.algos.remove(alg)
                    continue
                for k in range(c.K):
                    upstreamNodes, upstreamNeighs = get_upstream_nodes(G, k)
                    downstreamNodes, downstreamNeighs = get_downstream_nodes(G, k)
                    Cqk = [None for _ in range(c.K)]
                    if c.observability == 'poss':
                        hQkq = [[None for _ in range(c.K)] for _ in range(c.K)]
                        for q in range(c.K):
                            if q == k:
                                continue
                            obss = np.sum(
                                self.obsMat[np.array(list(upstreamNodes[q]) + [q]), :],
                                axis=0
                            ) > 0   # boolean vector indicating which sources are
                                    # observed by node q or any node upstream of q 
                            hQkq[k][q] = int(np.sum(self.obsMat[k, :] + obss == 2))
                            # ^^^ number of channels exchanged downstream by node q
                                # when node k is the root                    
                    for q in range(c.K):
                        if c.observability == 'foss':
                            dim = c.Mk + c.Q * len(upstreamNeighs[q])
                        elif c.observability == 'poss':
                            dim = c.Mk + int(
                                np.sum([hQkq[k][u] for u in upstreamNeighs[q]])
                            )
                        Cqk[q] = np.zeros((c.M, dim))
                        Cqk[q][c.Mk * q:c.Mk * (q + 1), :c.Mk] = np.eye(c.Mk)
                    # Compute fusion matrices
                    Pk = [None for _ in range(c.K)]
                    for q in flatten_list(tree_levels(G, k)):
                        for ii, n in enumerate(upstreamNeighs[q]):
                            if c.observability == 'foss':
                                idxBeg = c.Mk + ii * c.Q
                                idxEnd = idxBeg + c.Q
                            else:
                                idxBeg = c.Mk + int(
                                    np.sum([hQkq[k][u] for u in upstreamNeighs[q][:ii]])
                                )
                                idxEnd = idxBeg + hQkq[k][n]
                            Cqk[q][:, idxBeg:idxEnd] = Cqk[n] @ Pk[n]
                        if q != k:
                            # Compute Pk
                            Rhyqhyq = Cqk[q].T @ Ryy @ Cqk[q]
                            if c.observability == 'foss':
                                dim = c.Q
                            elif c.observability == 'poss':
                                dim = hQkq[k][q]
                            hEq = np.zeros((c.M, dim))
                            # hEq[c.Mk * k:c.Mk * k + dim, :dim] = np.eye(dim)
                            hEq[
                                c.Mk * downstreamNeighs[q]:\
                                c.Mk * downstreamNeighs[q] + dim, :dim
                            ] = np.eye(dim)
                            Rhyqyktq = Cqk[q].T @ Ryy @ hEq
                            Pk[q] = np.linalg.inv(Rhyqhyq) @ Rhyqyktq
                    # Compute estimation filter
                    tRyy = Cqk[k].T @ Ryy @ Cqk[k]
                    tRss = Cqk[k].T @ Rss @ Cqk[k]
                    W_netWide[alg][k] = Cqk[k] @ np.linalg.inv(tRyy) @ tRss[:, :c.D]
                    
                    # Compute metrics
                    if 'msew' in c.metricsToCompute:
                        metrics['msew'][alg][k] = np.mean(
                            np.abs(W_netWide[alg][k] - W_netWide['centralized'][k]) ** 2
                        )
                    if 'msed' in c.metricsToCompute:
                        dhatk = W_netWide[alg][k].T @ y
                        metrics['msed'][alg][k] = np.mean(np.abs(d[k, :] - dhatk) ** 2)
            elif 'danse' in alg:
                # Adapt metrics dimension
                for m in c.metricsToCompute:
                    metrics[m][alg] = np.zeros((c.K, c.maxDANSEiter))
                # Initialize the fusion matrices
                Pk = [randmat((c.Mk, c.Qd)) for _ in range(c.K)]
                WkkPrev = [np.zeros((c.Mk, c.Qd)) for _ in range(c.K)]
                u = 0  # updating node index
                for i in range(c.maxDANSEiter):
                    for k in range(c.K):
                        if alg.startswith("tidanse"):
                            Ck = np.zeros((c.M, c.Mk + c.Qd))
                            Ck[c.Mk * k:c.Mk * (k + 1), :c.Mk] = np.eye(c.Mk)
                            for q in range(c.K):
                                if q != k:
                                    Ck[c.Mk * q:c.Mk * (q + 1), c.Mk:] = Pk[q]
                        else:
                            Ck = np.zeros((c.M, c.Mk + c.Qd * (c.K - 1)))
                            Ck[c.Mk * k:c.Mk * (k + 1), :c.Mk] = np.eye(c.Mk)
                            idxNei = 0
                            for q in range(c.K):
                                if q != k:
                                    idxBeg = c.Mk + idxNei * c.Qd
                                    idxEnd = idxBeg + c.Qd
                                    Ck[c.Mk * q:c.Mk * (q + 1), idxBeg:idxEnd] = Pk[q]
                                    idxNei += 1
                        # Compute the filters
                        tRyy = Ck.T @ Ryy @ Ck
                        tRss = Ck.T @ Rss @ Ck
                        tW = np.linalg.pinv(tRyy) @ tRss
                        if alg.startswith("rsdanse"):
                            alpha = 1 / np.log10(i + 10)
                            tW[:c.Mk, :c.Qd] = (1 - alpha) * WkkPrev[k] + alpha * tW[:c.Mk, :c.Qd]
                            WkkPrev[k] = tW[:c.Mk, :c.Qd]
                        if k == u:
                            if alg.startswith("tidanse"):
                                Pk[u] = tW[:c.Mk, :c.Qd] @ np.linalg.pinv(tW[c.Mk:, :c.Qd])
                            else:
                                Pk[u] = tW[:c.Mk, :c.Qd]
                        W_netWide[alg][k] = Ck @ tW[:, :c.D]

                        # Compute metrics
                        if 'msew' in c.metricsToCompute:
                            metrics['msew'][alg][k, i] = np.mean(
                                np.abs(W_netWide[alg][k] - W_netWide['centralized'][k]) ** 2
                            )
                        if 'msed' in c.metricsToCompute:
                            dhatk = W_netWide[alg][k].T @ y
                            metrics['msed'][alg][k, i] = np.mean(np.abs(d[k, :] - dhatk) ** 2)
                    u = (u + 1) % c.K  # Update the node index for next iteration
            else:
                raise ValueError(f"Unknown algorithm: {alg}")
        
        fig, axes = plt.subplots(1, len(c.metricsToCompute))
        fig.set_size_inches(8.5, 3.5)
        for ii, m in enumerate(c.metricsToCompute):
            ax = axes[ii] if len(c.metricsToCompute) > 1 else axes
            if any('danse' in alg for alg in c.algos):
                ax.set_yscale('log')
                for ii, alg in enumerate(metrics[m].keys()):
                    if m == 'msew' and alg in ['centralized', 'local']:
                        continue
                    if 'danse' in alg:
                        if metrics[m][alg] is not None:
                            ax.plot(np.mean(metrics[m][alg], axis=0), label=alg)
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
        # for alg in c.algos:
        #     if alg == 'centralized':
        #         continue
        #     string = f"MSE_W {alg}: %10.3E" % np.mean(msew[alg])
        #     if c.scmEstimation == 'oracle':
        #         thrs = 1e-10
        #     elif c.scmEstimation == 'batch':
        #         thrs = 1e-4
        #     if np.mean(msew[alg]) < thrs:
        #         string += f" (PASSED [{c.scmEstimation}])"
        #     else:
        #         string += f" (FAILED [{c.scmEstimation}])"
        #     print(string)


def randmat(shape):
    """Generate a random matrix with given shape."""
    return np.random.rand(*shape)
    

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

    T = nx.Graph()

    # Step 1: create a path of length E (E+1 nodes)
    path_nodes = list(range(E + 1))
    for i in range(E):
        T.add_edge(path_nodes[i], path_nodes[i+1])

    next_node = E + 1

    # Step 2: attach remaining nodes to internal nodes on the path
    attachable_nodes = path_nodes[1:-1]  # avoid attaching to endpoints to preserve diameter

    i = 0
    while next_node < N:
        attach_to = attachable_nodes[i % len(attachable_nodes)]
        T.add_edge(attach_to, next_node)
        next_node += 1
        i += 1

    return T
