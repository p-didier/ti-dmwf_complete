# Contents of package:
# Classes and functions related to ....
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import numpy as np
import networkx as nx
import scipy.linalg as sla
from .base import Parameters
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Run:
    cfg: Parameters

    def launch(self):
        # Generate scenario
        c = self.cfg
        Aglob = self.randmat((c.M, c.Qdglob))
        Bglob = self.randmat((c.M, c.Qnglob))
        Aloc = sla.block_diag(*[self.randmat((c.Mk, c.Qdloc)) for _ in range(c.K)])
        Bloc = sla.block_diag(*[self.randmat((c.Mk, c.Qnloc)) for _ in range(c.K)])

        Rdgg = Aglob @ Aglob.T
        Rngg = Bglob @ Bglob.T
        Rdll = Aloc @ Aloc.T
        Rnll = Bloc @ Bloc.T
        Rss = Rdgg + Rdll
        Rnn = Rngg + Rnll
        Ryy = Rss + Rnn + np.eye(c.M) * 1e-6  # Add small self-noise to avoid singularity
        Rgg = Rdgg + Rngg
        print("Generated covariance matrices.")

        # Generate tree
        G = nx.complete_graph(c.K)
        for (u, v) in G.edges():
            G.edges[u, v]['weight'] = np.random.random()
        G = nx.minimum_spanning_tree(G)

        Wfilt = dict([(alg, [
            None for _ in range(c.K)
        ]) for alg in c.algos])  # Initialize node-speciifc filters dictionary
        for alg in c.algos:
            if alg == "centralized":
                Wcentr = np.linalg.inv(Ryy) @ Rss
                Wfilt[alg] = [
                    Wcentr[:, c.Mk * k:c.Mk * k + c.D]
                    for k in range(c.K)
                ]
            elif "idanse" in alg:
                # Fusion matrices
                Pk = [None for _ in range(c.K)]
                for k in range(c.K):
                    Rykyk = Ryy[c.Mk * k:c.Mk * (k + 1), c.Mk * k:c.Mk * (k + 1)]
                    Rgkgk = Rgg[c.Mk * k:c.Mk * (k + 1), c.Mk * k:c.Mk * (k + 1)]
                    Pk[k] = np.linalg.inv(Rykyk) @ Rgkgk[:, :c.Qe]
                # Estimation filters
                for k in range(c.K):
                    # ty = C^H.y
                    if alg == "idanse":
                        Ck = np.zeros((c.M, c.Mk + c.Qe * (c.K - 1)))
                        Ck[c.Mk * k:c.Mk * (k + 1), :c.Mk] = np.eye(c.Mk)
                        idxNei = 0
                        for q in range(c.K):
                            if q != k:
                                Ck[
                                    c.Mk * q:c.Mk * (q + 1),
                                    c.Mk + idxNei * c.Qe: c.Mk + (idxNei + 1) * c.Qe
                                ] = Pk[q]
                                idxNei += 1
                    elif alg == "tiidanse":
                        Ck = np.zeros((c.M, c.Mk + c.Qe))
                        Ck[c.Mk * k:c.Mk * (k + 1), :c.Mk] = np.eye(c.Mk)
                        for q in range(c.K):
                            if q != k:
                                Ck[c.Mk * q:c.Mk * (q + 1), c.Mk:] = Pk[q]
                    # Compute the filters
                    tRyy = Ck.T @ Ryy @ Ck
                    tRss = Ck.T @ Rss @ Ck
                    Wfilt[alg][k] = Ck @ np.linalg.inv(tRyy) @ tRss[:, :c.D]
            elif alg == "tidmwf":
                for k in range(c.K):
                    _, upstreamNeighs = get_upstream_nodes(G, k)
                    Cqk = [
                        np.zeros((c.M, c.Mk + c.Qe * len(upstreamNeighs[q])))
                        for q in range(c.K)
                    ]
                    for q in flatten_list(tree_levels(G, k)):
                        Cqk[q][c.Mk * q:c.Mk * (q + 1), :c.Mk] = np.eye(c.Mk)
                        for ii, n in enumerate(upstreamNeighs[q]):
                            Cqk[q][
                                :, c.Mk + ii * c.Qe: c.Mk + (ii + 1) * c.Qe
                            ] = Cqk[n] @ Pk[n]
                        if q != k:
                            # Compute Pk
                            Rhyhy = Cqk[q].T @ Ryy @ Cqk[q]
                            Rhghg = Cqk[q].T @ Rgg @ Cqk[q]
                            Pk[q] = np.linalg.inv(Rhyhy) @ Rhghg[:, :c.Qe]
                    # Compute estimation filter
                    tRyy = Cqk[k].T @ Ryy @ Cqk[k]
                    tRss = Cqk[k].T @ Rss @ Cqk[k]
                    Wfilt[alg][k] = Cqk[k] @ np.linalg.inv(tRyy) @ tRss[:, :c.D]
            else:
                raise ValueError(f"Unknown algorithm: {alg}")
        
        if 'centralized' in c.algos:
            # Compute MSE_W
            msew = dict([(alg, np.zeros(c.K)) for alg in c.algos])
            for alg in c.algos:
                for k in range(c.K):
                    msew[alg][k] = np.mean(
                        np.abs(Wfilt[alg][k] - Wfilt['centralized'][k]) ** 2
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
            
            print("MSE of filters:")
            for alg in c.algos:
                if alg == 'centralized':
                    continue
                print(f"{alg}: {np.mean(msew[alg]):.10f} ± {np.std(msew[alg]):.10f}")

    def randmat(self, shape):
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