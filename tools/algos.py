# Contents of package:
# Classes and functions related to ....
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

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
    Qkq: np.ndarray = None  # Number of sources in common between nodes

    def setup_scms(self):
        c = self.cfg

        # Compute steering matrices
        if c.observability == 'foss':
            self.Qkq = np.full((c.K, c.K), c.Q)
            Amat = randmat((c.M, c.Qd))
            Bmat = randmat((c.M, c.Qn))
        elif c.observability == 'poss':
            # Do not differentiate between global and local sources, 
            # randomly generate observability pattern
            self.obsMat = np.zeros((c.K, c.Q))
            def inadequate(om):
                return np.any(np.sum(om, axis=1) == 0) | np.any(np.sum(om, axis=0) == 0)
            # Criterion for adequacy: at least one desired source and one noise
            # source must be observed by each node, and each source must be
            # observed by at least one node.
            while inadequate(self.obsMat[:, :c.Qd]) or inadequate(self.obsMat[:, c.Qd:]):
                self.obsMat = np.random.randint(0, 2, (c.K, c.Q))
            if c.possDiffuse:
                # Make sure each noise source is only observed by one node at most
                for n in range(c.Qn):
                    idx = np.where(self.obsMat[:, c.Qd + n] == 1)[0]
                    if len(idx) > 1:
                        idx = np.random.choice(idx, 1, replace=False)
                        self.obsMat[:, c.Qd + n] = 0
                        self.obsMat[idx, c.Qd + n] = 1
            Amat = randmat((c.M, c.Qd))
            Bmat = randmat((c.M, c.Qn))
            for k in range(c.K):
                for s in range(c.Qd):
                    if self.obsMat[k, s] == 0:
                        Amat[c.Mk * k:c.Mk * (k + 1), s] = 0
                for n in range(c.Qn):
                    if self.obsMat[k, c.Qd + n] == 0:
                        Bmat[c.Mk * k:c.Mk * (k + 1), n] = 0
            # Number of sources in common between node k and q
            self.Qkq = np.zeros((c.K, c.K), dtype=int)
            for k in range(c.K):
                for q in range(c.K):
                    if k == q:
                        continue
                    # self.Qkq[k, q] = np.sum(self.obsMat[k, :] & self.obsMat[q, :])
                    self.Qkq[k, q] = np.sum(self.obsMat[k, :])
            # Qkq should be symmetric
            # assert np.all(self.Qkq == self.Qkq.T), "Qkq should be symmetric"
        
        # Compute the SCMs
        if c.scmEstimation == 'oracle':
            # For oracle SCM estimation, we assume perfect knowledge of the
            # source and noise steering matrices
            Rss = Amat @ Amat.T
            Rnn = Bmat @ Bmat.T
            Rvv = np.eye(c.M) * c.selfNoiseFactor  # small self-noise
        elif c.scmEstimation == 'batch':
            # Batch SCM estimation based on actual signals
            slat = np.random.randn(c.Qd, c.N)
            nlat = np.random.randn(c.Qn, c.N)
            s = Amat @ slat
            n = Bmat @ nlat
            v = np.random.randn(c.M, c.N) * c.selfNoiseFactor  # small self-noise
            Rss = s @ s.T / c.N
            Rnn = n @ n.T / c.N
            Rvv = v @ v.T / c.N
        
        # Complete signa SCM
        Ryy = Rss + Rnn + Rvv

        return Ryy, Rss, Rnn, Rvv, Amat, Bmat

    def launch(self):
        # Generate scenario
        c = self.cfg
        Ryy, Rss, Rnn, Rvv, Amat, Bmat = self.setup_scms()
        
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
        for alg in c.algos:
            if alg == "centralized":
                Wcentr = np.linalg.inv(Ryy) @ Rss
                W_netWide[alg] = [
                    Wcentr[:, c.Mk * k:c.Mk * k + c.D]
                    for k in range(c.K)
                ]

                # DEBUG -- testing math from Appendix A in paper (dMWF proof)
                # /!\ Assuming k = 0
                Gam = np.linalg.inv(Rvv)
                Cmat = np.hstack([Amat, Bmat])
                cRlat = np.eye(c.Q)  # latent signals with unit power
                Xmat = np.linalg.inv(cRlat) + Cmat.T @ Gam @ Cmat
                RyyInv = Gam - Gam @ Cmat @ np.linalg.inv(Xmat) @ Cmat.T @ Gam
                Rlatext = np.diag(np.hstack((np.ones(c.Qd), np.zeros(c.Qn))))
                Mmat = (np.eye(c.Q) - np.linalg.inv(Xmat) @ Cmat.T @ Gam @ Cmat) @ Rlatext @ Cmat.T
                Mmat = Mmat[:, :c.Mk]  # Keep only the first c.D columns
                hW = Gam @ Cmat @ Mmat
                hW_comp = np.zeros_like(hW)
                for q in range(c.K):
                    Cmatq = Cmat[c.Mk * q:c.Mk * (q + 1), :]
                    # Extract columns of Cmatq corresponding to sources observed
                    # by both q and k
                    idx = np.where(self.obsMat[q, :] & self.obsMat[0, :])[0]
                    Cmatq = Cmatq[:, idx]
                    Mmatq = Mmat[idx, :]
                    Gamq = Gam[c.Mk * q:c.Mk * (q + 1), c.Mk * q:c.Mk * (q + 1)]
                    hW_comp[c.Mk * q:c.Mk * (q + 1), :] = Gamq @ Cmatq @ Mmatq

                if 0:  # test plot
                    fig, axes = plt.subplots(3, 3)
                    fig.set_size_inches(8.5, 6.5)
                    axes[0, 0].imshow(np.abs(RyyInv))
                    axes[0, 0].set_title("Ryy^-1 (MIL)")
                    axes[0, 1].imshow(np.abs(np.linalg.inv(Ryy)))
                    axes[0, 1].set_title("Ryy^-1 (Numpy)")
                    mapp = axes[0, 2].imshow(np.abs(RyyInv - np.linalg.inv(Ryy)))
                    axes[0, 2].set_title("Difference")
                    fig.colorbar(mapp, ax=axes[0, 2])
                    #
                    axes[1, 0].imshow(hW)
                    axes[1, 0].set_title("hW (MIL)")
                    axes[1, 1].imshow(Wcentr)
                    axes[1, 1].set_title("Wcentr (MIL)")
                    mapp = axes[1, 2].imshow(np.abs(hW - Wcentr))
                    axes[1, 2].set_title("Difference")
                    fig.colorbar(mapp, ax=axes[1, 2])
                    #
                    axes[2, 0].imshow(hW_comp)
                    axes[2, 0].set_title("hW (MIL)")
                    axes[2, 1].imshow(Wcentr)
                    axes[2, 1].set_title("Wcentr (MIL)")
                    mapp = axes[2, 2].imshow(np.abs(hW_comp - Wcentr))
                    axes[2, 2].set_title("Difference")
                    fig.colorbar(mapp, ax=axes[2, 2])
                    fig.tight_layout()
                    plt.show()
            elif alg == "dmwf":
                # Neighbor-specific fusion matrices
                Pk = [[None for _ in range(c.K)] for _ in range(c.K)]
                for q in range(c.K):
                    Ryqyq = Ryy[c.Mk * q:c.Mk * (q + 1), c.Mk * q:c.Mk * (q + 1)]
                    for k in range(c.K):
                        if k == q:
                            continue
                        Ryqyktq = Ryy[c.Mk * q:c.Mk * (q + 1), c.Mk * k:c.Mk * k + self.Qkq[q][k]]
                        Pk[q][k] = np.linalg.inv(Ryqyq) @ Ryqyktq
                # Estimation filters
                for k in range(c.K):
                    # ty = C^H.y
                    if c.observability == 'foss':
                        Ck = np.zeros((c.M, c.Mk + c.Q * (c.K - 1)))
                    elif c.observability == 'poss':
                        Ck = np.zeros((c.M, c.Mk + int(np.sum(self.Qkq[:, k]))))
                    Ck[c.Mk * k:c.Mk * (k + 1), :c.Mk] = np.eye(c.Mk)
                    idxNei = 0
                    # QkqNeighs = self.Qkq[k, :]
                    # QkqNeighs = np.delete(QkqNeighs, k)  # Remove k
                    idxEnd = c.Mk
                    for q in range(c.K):
                        if q != k:
                            if c.observability == 'foss':
                                idxBeg = c.Mk + idxNei * c.Q
                                if idxBeg != idxEnd:
                                    raise ValueError("Indexing error in dMWF fusion matrix.")
                                idxEnd = idxBeg + c.Q
                            elif c.observability == 'poss':
                                idxBeg = c.Mk + int(np.sum(self.Qkq[:q, k]))
                                if idxBeg != idxEnd:
                                    raise ValueError("Indexing error in dMWF fusion matrix.")
                                idxEnd = idxBeg + self.Qkq[q][k]
                            Ck[c.Mk * q:c.Mk * (q + 1), idxBeg:idxEnd] = Pk[q][k]
                            idxNei += 1
                    # Compute the filters
                    tRyy = Ck.T @ Ryy @ Ck
                    tRss = Ck.T @ Rss @ Ck
                    tW = np.linalg.inv(tRyy) @ tRss[:, :c.D]
                    W_netWide[alg][k] = Ck @ tW

                    # # DEBUG -- testing math from Appendix A in paper (dMWF proof)
                    # W_netWide_theo = np.zeros_like(W_netWide[alg][k])
                    # Pqk = [None for _ in range(c.K)]
                    # for q in range(c.K):
                    #     Rlatkq = np.eye(self.Qkq[k, q])  # latent signals with unit power
                    #     Cmatq = Cmat[c.Mk * q:c.Mk * (q + 1), :]
                    #     non_zero_idx = np.where(np.any(Cmatq != 0, axis=0))[0]
                    #     Cmatq = Cmatq[:, non_zero_idx]
                    #     Gamq = Gam[c.Mk * q:c.Mk * (q + 1), c.Mk * q:c.Mk * (q + 1)]
                    #     Xmatq = np.linalg.inv(Rlatkq) + Cmatq.T @ Gamq @ Cmatq
                    #     Mqk = (np.eye(self.Qkq[k, q]) - np.linalg.inv(Xmatq) @ Cmatq.T @ Gamq @ Cmatq) @\
                    #         Rlatkq @ Cmatq.T
                    #     Pqk[q] = Gamq @ Cmatq @ Mqk[:, :self.Qkq[k, q]]
                    #     if q != k:
                    #         # Extract Gkq matrix from tW
                    #         idxBeg = c.Mk + np.sum(self.Qkq[k, :q])
                    #         idxEnd = idxBeg + self.Qkq[k][q]
                    #         Gkq = tW[idxBeg:idxEnd, :]
                    #         W_netWide_theo[c.Mk * q:c.Mk * (q + 1), :] = Pqk[q] @ Gkq
                    #     else:
                    #         W_netWide_theo[c.Mk * k:c.Mk * (k + 1), :] = tW[:c.Mk, :]
                    
                    # if 0:  # test plot
                    #     fig, axes = plt.subplots(1, 3)
                    #     fig.set_size_inches(8.5, 3.5)
                    #     axes[0].imshow(W_netWide_theo)
                    #     axes[0].set_title("Ryy^-1 (MIL)")
                    #     axes[1].imshow(W_netWide[alg][k])
                    #     axes[1].set_title("Ryy^-1 (Numpy)")
                    #     mapp = axes[2].imshow(np.abs(W_netWide[alg][k] - W_netWide_theo))
                    #     axes[2].set_title("Difference")
                    #     fig.colorbar(mapp, ax=axes[2])
                    #     pass
                    
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
            else:
                raise ValueError(f"Unknown algorithm: {alg}")
        
        if 'centralized' in c.algos:
            # Compute MSE_W
            msew = dict([(alg, np.zeros(c.K)) for alg in c.algos])
            for alg in c.algos:
                for k in range(c.K):
                    msew[alg][k] = np.mean(
                        np.abs(W_netWide[alg][k] - W_netWide['centralized'][k]) ** 2
                    )

            if 0:
                fig, axes = plt.subplots(1, 1)
                fig.set_size_inches(3.5, 3.5)
                idx = 0
                for alg in c.algos:
                    if alg == 'centralized':
                        continue
                    toPlot = np.mean(msew[alg])
                    axes.bar(idx, toPlot, yerr=np.std(msew[alg]), label=alg)
                    idx += 1
                axes.set_xticks(range(len(c.algos) - 1))
                axes.set_xticklabels([alg for alg in c.algos if alg != 'centralized'])
                axes.set_ylabel("MSE of filters")
                axes.grid()
                fig.tight_layout()
                plt.show()
            
            for alg in c.algos:
                if alg == 'centralized':
                    continue
                string = f"MSE_W {alg}: {np.mean(msew[alg]):.10f} ± {np.std(msew[alg]):.10f}"
                if np.mean(msew[alg]) < 1e-10:
                    string += " (PASSED)"
                else:
                    string += " (FAILED)"
                print(string)

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
