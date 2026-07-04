import networkx as nx
import pandas as pd
import os
from typing import Dict, List


class SingleDstPath(Dict):
    partition: int
    edges: List[List]  # [[src, dst, edge data]]


class BroadCastTopology:
    def __init__(self, src: str, dsts: List[str], num_partitions: int = 4, paths: Dict[str, 'SingleDstPath'] = None):
        self.src = src
        self.dsts = dsts
        self.num_partitions = num_partitions
        if paths is not None:
            self.paths = paths
        else:
            self.paths = {dst: {str(i): None for i in range(num_partitions)} for dst in dsts}

    def get_paths(self):
        return self.paths

    def set_num_partitions(self, num_partitions: int):
        self.num_partitions = num_partitions

    def set_dst_partition_paths(self, dst: str, partition: int, paths: List[List]):
        partition = str(partition)
        self.paths[dst][partition] = paths

    def append_dst_partition_path(self, dst: str, partition: int, path: List):
        partition = str(partition)
        if self.paths[dst][partition] is None:
            self.paths[dst][partition] = []
        self.paths[dst][partition].append(path)


def search_algorithm(src, dsts, G, num_partitions):
    """
    Find broadcast paths from source to all destinations.
    
    Uses a Directed Steiner Tree heuristic (Takahashi-Matsuyama) to minimize total egress cost,
    while applying a load-based penalty to encourage multipath routing across partitions 
    to balance throughput and reduce transfer time.
    """
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    # Initialize edge loads to track usage across partitions
    for u, v, d in h.edges(data=True):
        d['load'] = 0

    for p in range(num_partitions):
        # Update edge weights: base cost + penalty for load + penalty for low throughput
        for u, v, d in h.edges(data=True):
            c = d.get('cost', 0.0)
            t = d.get('throughput', 1.0)
            # Heavily weight cost to minimize egress fees, while using throughput and load for tie-breaking and multipath
            d['weight'] = (c * 1000.0) + (5.0 / max(t, 0.1)) + (10.0 * d['load'] / max(t, 0.1))

        tree_nodes = {src}
        remaining_dsts = set(dsts)
        parent = {}
        
        # Greedily grow the broadcast tree
        while remaining_dsts:
            dummy = 'DUMMY_SRC'
            h.add_node(dummy)
            for tn in tree_nodes:
                h.add_edge(dummy, tn, weight=0)
                
            try:
                lengths, paths = nx.single_source_dijkstra(h, dummy, weight='weight')
            except nx.NetworkXNoPath:
                lengths, paths = {}, {}
                
            h.remove_node(dummy)
            
            best_dist = float('inf')
            best_path = None
            best_dst = None
            
            for r_dst in remaining_dsts:
                if r_dst in lengths and lengths[r_dst] < best_dist:
                    best_dist = lengths[r_dst]
                    best_path = paths[r_dst][1:]  # Exclude dummy node
                    best_dst = r_dst
                    
            if best_path is None:
                break  # Disconnected
                
            # Add the chosen path to the tree
            for i in range(len(best_path) - 1):
                u, v = best_path[i], best_path[i + 1]
                if v not in parent:
                    parent[v] = u
                    tree_nodes.add(v)
                    h[u][v]['load'] += 1
            
            remaining_dsts.remove(best_dst)
            
        # Extract paths from the constructed tree for each destination
        for dst in dsts:
            if dst not in parent:
                # Fallback to direct shortest path if heuristic fails to reach dst
                try:
                    path = nx.dijkstra_path(G, src, dst, weight='cost')
                    edges = [[path[i], path[i+1], G[path[i]][path[i+1]]] for i in range(len(path)-1)]
                    bc_topology.set_dst_partition_paths(dst, p, edges)
                except nx.NetworkXNoPath:
                    pass
                continue
                
            curr = dst
            path_edges = []
            while curr != src:
                p_node = parent[curr]
                path_edges.append([p_node, curr, G[p_node][curr]])
                curr = p_node
            path_edges.reverse()
            bc_topology.set_dst_partition_paths(dst, p, path_edges)

    return bc_topology