#!pip install networkx matplotlib numpy cvxpy

import time
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp   # For LP-relaxation

# Algorithms

def greedy_vertex_cover(G):
    """Greedy 2-approximation algorithm"""
    start = time.time()
    cover = set()
    edges = set(G.edges())
    while edges:
        u, v = edges.pop()
        cover.add(u)
        cover.add(v)
        edges = {e for e in edges if u not in e and v not in e}
    return cover, time.time() - start


def lp_relaxation_vertex_cover(G):
    """Linear Programming relaxation for Vertex Cover"""
    start = time.time()
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    index = {nodes[i]: i for i in range(n)}

    x = cp.Variable(n)
    constraints = [x[index[u]] + x[index[v]] >= 1 for u, v in G.edges()]
    constraints += [x >= 0, x <= 1]
    objective = cp.Minimize(cp.sum(x))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver="SCS", verbose=False)

    cover = {nodes[i] for i in range(n) if x.value[i] >= 0.5}
    return cover, time.time() - start


def randomized_vertex_cover(G):
    """Randomized heuristic algorithm"""
    start = time.time()
    edges = set(G.edges())
    cover = set()
    while edges:
        u, v = random.choice(list(edges))
        chosen = random.choice([u, v])
        cover.add(chosen)
        edges = {e for e in edges if chosen not in e}
    return cover, time.time() - start


def coverage_fraction(G, cover):
    """Fraction of edges covered"""
    if G.number_of_edges() == 0:
        return 1.0
    covered = sum(1 for (u, v) in G.edges() if u in cover or v in cover)
    return covered / G.number_of_edges()

# Experiments

def run_experiments(node_sizes=[500], densities=[0.002, 0.005, 0.01], trials=5):
    results = []
    for n in node_sizes:
        for p in densities:
            G = nx.erdos_renyi_graph(n=n, p=p, seed=42)

            # Greedy
            g_cover, g_time = greedy_vertex_cover(G)

            # LP-relaxation
            lp_cover, lp_time = lp_relaxation_vertex_cover(G)

            # Randomized (average of trials)
            r_sizes, r_times = [], []
            for _ in range(trials):
                r_cover, r_time = randomized_vertex_cover(G)
                r_sizes.append(len(r_cover))
                r_times.append(r_time)

            results.append({
                "n": n,
                "p": p,
                "Greedy": (len(g_cover), g_time, coverage_fraction(G, g_cover)),
                "LP": (len(lp_cover), lp_time, coverage_fraction(G, lp_cover)),
                "Randomized": (np.mean(r_sizes), np.mean(r_times), 1.0)
            })

            print(f"Graph n={n}, p={p:.3f} -> Done")

    return results

# Visualization with log scale for time

def plot_results(results):
    densities = sorted(set(r["p"] for r in results))
    algos = ["Greedy", "LP", "Randomized"]

    for n in sorted(set(r["n"] for r in results)):
        # Cover Size
        plt.figure(figsize=(8,4))
        for algo in algos:
            sizes = [r[algo][0] for r in results if r["n"] == n]
            plt.plot(densities, sizes, marker='o', label=algo)
        plt.title(f"Cover Size vs Graph Density (n={n})")
        plt.xlabel("Edge Density (p)")
        plt.ylabel("Number of Selected Sensors")
        plt.grid(True)
        plt.legend()
        plt.show()

        # Execution Time (log scale)
        plt.figure(figsize=(8,4))
        for algo in algos:
            times = [r[algo][1] for r in results if r["n"] == n]
            plt.plot(densities, times, marker='s', label=algo)
        plt.yscale("log")  # Makes small times visible
        plt.title(f"Execution Time vs Graph Density (n={n})")
        plt.xlabel("Edge Density (p)")
        plt.ylabel("Time (seconds, log scale)")
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.show()

# Main
if __name__ == "__main__":
    results = run_experiments()
    plot_results(results)