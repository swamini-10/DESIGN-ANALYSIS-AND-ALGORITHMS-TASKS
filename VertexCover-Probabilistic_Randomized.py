!pip install networkx pandas matplotlib --quiet

import time
import random
import multiprocessing as mp
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Randomized vertex cover
def randomized_vertex_cover(G, rng=None):
    """Randomized algorithm for vertex cover."""
    if rng is None:
        rng = random
    cover = set()
    remaining_edges = set(G.edges())
    incident = {v: set() for v in G.nodes()}
    for u, v in list(remaining_edges):
        e = (u, v)
        incident[u].add(e)
        incident[v].add(e)

    while remaining_edges:
        e = rng.choice(tuple(remaining_edges))
        u, v = e
        chosen = rng.choice((u, v))
        cover.add(chosen)
        for inc in list(incident[chosen]):
            if inc in remaining_edges:
                remaining_edges.discard(inc)
            a, b = inc
            other = b if a == chosen else a
            if inc in incident[other]:
                incident[other].discard(inc)
        incident[chosen].clear()
    return cover

# Deterministic 2-approximation

def deterministic_matching_cover(G):
    """Deterministic 2-approximation: maximal matching."""
    matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True)
    cover = set()
    for u, v in matching:
        cover.add(u)
        cover.add(v)
    return cover

# Single trial run

def single_trial(n, p, rng_seed=None):
    rng = random.Random(rng_seed)
    G = nx.fast_gnp_random_graph(n, p, seed=rng_seed)
    t0 = time.perf_counter()
    cover_rand = randomized_vertex_cover(G, rng)
    t1 = time.perf_counter()
    cover_det = deterministic_matching_cover(G)
    t2 = time.perf_counter()
    return {
        'n': n,
        'p': p,
        'rand_size': len(cover_rand),
        'det_size': len(cover_det),
        'rand_time': t1 - t0,
        'det_time': t2 - t1
    }

def run_experiments(n=200, densities=None, trials=10, processes=1):
    if densities is None:
        densities = [0.001, 0.005, 0.01, 0.02, 0.05]
    results = []

    logging.info("Running experiments...")
    if processes is None or processes <= 1:
        for p in densities:
            for i in range(trials):
                seed = int(time.time() * 1000) ^ (i + int(p * 1e6))
                results.append(single_trial(n, p, rng_seed=seed))
    else:
        pool = mp.Pool(processes=processes)
        try:
            tasks = []
            for p in densities:
                for i in range(trials):
                    seed = (i + 1) + int(p * 1e6)
                    tasks.append((n, p, seed))
            func = lambda args: single_trial(*args)
            for res in pool.imap_unordered(func, tasks):
                results.append(res)
        finally:
            pool.close()
            pool.join()

    df = pd.DataFrame(results)
    agg = df.groupby('p').agg(
        avg_rand_size=('rand_size', 'mean'),
        var_rand_size=('rand_size', 'var'),
        avg_det_size=('det_size', 'mean'),
        var_det_size=('det_size', 'var'),
        avg_rand_time=('rand_time', 'mean'),
        avg_det_time=('det_time', 'mean')
    ).reset_index()

    df['matching_size'] = (df['det_size'] / 2).replace(0, np.nan)
    df['rand_over_matching'] = df['rand_size'] / df['matching_size']
    emp_ratio = df.groupby('p')['rand_over_matching'].mean().reset_index(name='emp_avg_rand_over_matching')

    #Plots
    plt.figure(figsize=(8, 5))
    plt.plot(agg['p'], agg['avg_rand_size'], marker='o', label='Randomized avg size')
    plt.plot(agg['p'], agg['avg_det_size'], marker='o', label='Deterministic avg size')
    plt.xscale('log')
    plt.xlabel('Graph density p (log scale)')
    plt.ylabel('Average vertex cover size')
    plt.title(f'Average Vertex Cover Size vs Density (n={n}, trials={trials})')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(agg['p'], agg['var_rand_size'], marker='o')
    plt.xscale('log')
    plt.xlabel('Graph density p (log scale)')
    plt.ylabel('Variance of randomized cover size')
    plt.title(f'Variance of Randomized Vertex Cover Size vs Density (n={n}, trials={trials})')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(agg['p'], agg['avg_rand_time'], marker='o', label='Randomized avg time (s)')
    plt.plot(agg['p'], agg['avg_det_time'], marker='o', label='Deterministic avg time (s)')
    plt.xscale('log')
    plt.xlabel('Graph density p (log scale)')
    plt.ylabel('Average execution time (s)')
    plt.title(f'Execution Time vs Density (n={n}, trials={trials})')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    return df, agg, emp_ratio

df_all, agg, emp_ratio = run_experiments(n=200, densities=[0.001, 0.005, 0.01, 0.02, 0.05], trials=10, processes=1)

print("\nAggregated Results:\n", agg)
print("\nEmpirical Approximation Ratio (randomized / matching size):\n", emp_ratio)