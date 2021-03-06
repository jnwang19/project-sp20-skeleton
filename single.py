import networkx as nx
from networkx.algorithms import approximation
from collections import deque
import numpy as np
from parse import read_input_file, write_output_file, read_output_file
from utils import is_valid_network, average_pairwise_distance_fast
import sys
import os
import random
import multiprocessing
import time
from itertools import combinations

# global variables
# dictionary that holds each id, input networkx.Graph
INPUT_PATH = '../inputs/'
OUTPUT_PATH = '../outputs-small-28/'
FINISHED_FILE_PATH = 'finished_files.txt'
METHODS_PATH = '../methods.txt'

inputs = {}
# dictionary that holds id, score
best_scores = {}
best_methods = {}
max_weight = {}

finished_files = set()


def setup(inputs, best_scores, best_methods, finished_files):
    finished_file = open(FINISHED_FILE_PATH, 'r')
    for file in finished_file:
        finished_files.add(file.split('\n')[0])
    for filename in os.listdir(INPUT_PATH):
        out_filename = filename.split('.')[0] + '.out'
        if filename not in finished_files:
            inputs[filename] = read_input_file(INPUT_PATH + filename)
            max_edge_weight = 0
            for edge in inputs[filename].edges:
                max_edge_weight = max(max_edge_weight, inputs[filename].get_edge_data(edge[0], edge[1])['weight'])
            max_weight[filename] = max_edge_weight
            if os.path.isfile(OUTPUT_PATH + out_filename):
                output_graph = read_output_file(OUTPUT_PATH + out_filename, inputs[filename])
                if output_graph.number_of_nodes() == 1:
                    best_scores[filename] = 0
                else:
                    best_scores[filename] = average_pairwise_distance_fast(output_graph)
            else:
                best_scores[filename] = float('inf')

def solve():
    """
    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph
    """

    # TODO: your code here!
    setup(inputs, best_scores, best_methods, finished_files)

    # run MST
    # for id in inputs:
    #     G = inputs[id]
    #     mst(G.copy(), id)
    #     mds(G.copy(), id)
    print("number of cpu: ", multiprocessing.cpu_count())
    #start = time.perf_counter()
    i = 0
    while(True):
        if i % 1000 == 0:
            print(i)
            print(best_scores['small-82.in'])
        #processes = []
        G = inputs['small-82.in']
        #mds(G.copy(), 'small-82.in')
        #mst(G.copy(), 'small-82.in')
        random_dom_set(G.copy(), 'small-82.in')
        #brute_force(G.copy(), 'small-28.in')
        #random_mds(G.copy(), 'small-82.in')
        bfs(G.copy(), 'small-82.in')
        #random_edges(G, 'small-28.in')
        #random_weight(G.copy(), 'small-28.in')
        #     p = multiprocessing.Process(target=bfs, args=[inputs[id].copy(), id])
        #     p.start()
        #     processes.append(p)
        # for process in processes:
        #     process.join()
        i += 1
        #write_best_methods()
    
    #write_finished_files()
    #finish = time.perf_counter()
    #print(f'time: {round(finish-start, 2)} seconds')
    write_best_methods()

    # # run everything else
    # while(True):
    #   write_best_methods()
    #     for id in inputs:
    #         G = inputs[id]
    #         bfs(G.copy(), id)
    #         random_mds(G.copy(), id)

# MST - Kruskal's + pruning- 1x
# network.MDS + network.Steiner
# BFS + pruning - multiple
# our MDS with steiner- multiple

def random_dom_set(G, id):
    T = nx.Graph()
    nodes = np.random.permutation(list(G.nodes))
    for node in nodes:
        T.add_node(node)
        if nx.is_dominating_set(G, T.nodes):
            break

    center = np.random.choice(list(T.nodes))

    shortest_paths = nx.shortest_path(G)
    #print(shortest_paths)
    for node in list(T.nodes):
        v = center
        for w in shortest_paths[center][node]:
            #print(v, w)
            if v != w:
                T.add_edge(v, w, weight=G.get_edge_data(v, w)['weight'])
                v = w

    leaves = find_leaves(T)
    T = prune(T, G, leaves)
    #print(average_pairwise_distance_fast(T))
    update_best_graph(T, id, 'dijkstra')


def find_leaves(T):
    num_nodes = T.number_of_nodes()
    leaves = deque()
    for v in T:
        neighbors = T[v]
        if len(neighbors) == 1 and len(leaves) + 1 < num_nodes:
            leaves.append(v)
    return leaves

def prune(T, G, leaves):
    score = score = average_pairwise_distance_fast(T)

    while leaves:
        l = leaves.popleft()
        parent = list(T.neighbors(l))[0]
        edge_weight = T.get_edge_data(parent, l)['weight']
        T.remove_node(l)
        if nx.is_dominating_set(G, T.nodes):
            new_score = average_pairwise_distance_fast(T)
            p = np.random.random()
            if new_score > score and p <0.5:
                T.add_node(l)
                T.add_edge(l, parent, weight=edge_weight)
            else:
                score = new_score
                #append new leaves
                if T.degree(parent) == 1:
                    leaves.append(parent)
        else:
            T.add_node(l)
            T.add_edge(l, parent, weight=edge_weight)

    #print('hello')
    #print(nx.is_tree(T))
    return T


def brute_force(G, id):
    edges = list(G.edges)
    for i in range(1, len(edges)):
        print(i)
        lst = list(combinations(edges, i))
        print(len(lst))
        for edge_group in lst:
            T = nx.Graph()
            for edge in edge_group:
                T.add_edge(edge[0], edge[1], weight=G.get_edge_data(edge[0], edge[1])['weight'])
            if is_valid_network(G, T) and average_pairwise_distance_fast(T) < best_scores[id]:
                print(average_pairwise_distance_fast(T))
                update_best_graph(T, id, 'brute_force')


def random_edges(G, id):
    T = nx.Graph()
    edges = list(np.random.permutation(G.edges))
    for edge in edges:
        T.add_edge(edge[0], edge[1], weight=G.get_edge_data(edge[0], edge[1])['weight'])
        if not nx.is_tree(T):
            T.remove_edge(edge[0], edge[1])
        if is_valid_network(G, T):
            score = average_pairwise_distance_fast(T)
            print(score - best_scores[id])
            update_best_graph(T, id, 'random edges')
            break

def random_weight(G, id):
    T = nx.Graph()
    nodes = list(G.nodes)
    v = np.random.choice(nodes, 1)[0]
    T.add_node(v)
    all_neighbors = set()
    count = 0
    flag = True
    while not is_valid_network(G, T):
        #print(count)
        if flag:
            neighbors = []
            for node in T:
                for w in G.neighbors(v):
                    if w not in T:
                        neighbors.append((node, w, 100 - G[v][w]['weight']))
            for n in neighbors:
                all_neighbors.add(n)

        total = sum([weight for u, w, weight in all_neighbors])
        dist = [weight / total for u, w, weight in all_neighbors]
        neighbor_edges = [(u, w, weight) for u, w, weight in all_neighbors]
        
        #print(dist)
        next_edge = neighbor_edges[np.random.choice(range(len(neighbor_edges)), p=dist)]
        T.add_edge(next_edge[0], next_edge[1], weight=next_edge[2])

        if nx.is_tree(T):
            v = next_edge[1]
            flag = True
        else:
            T.remove_edge(next_edge[0], next_edge[1])
            all_neighbors.remove(next_edge)
            flag = False

        count += 1
        if T.number_of_nodes() == G.number_of_nodes():
            print('no')
            break
    
    score = average_pairwise_distance_fast(T)

    T_copy = T.copy()
    for v in T:
        for w in G.neighbors(v):
            if w not in G:
                T_copy.add_edge(v, w, G.get_edge_data(v, w)['weight'])
                new_score = average_pairwise_distance_fast(T_copy)
                if new_score > score:
                    T_copy.remove_edge(v, w)
                else:
                    score = new_score
                    print('good')

    #print(score - best_scores[id])

    update_best_graph(T, id, 'random weight')
    




def mst(G, id):
    T_k = nx.minimum_spanning_tree(G, algorithm = 'kruskal')
    T_p = nx.minimum_spanning_tree(G, algorithm = 'prim')
    T_b = nx.minimum_spanning_tree(G, algorithm = 'boruvka')

    leaves_k = find_leaves(T_k)
    leaves_p = find_leaves(T_p)
    leaves_b = find_leaves(T_b)

    T_k = prune(T_k, G, leaves_k)
    T_p = prune(T_p, G, leaves_p)
    T_b = prune(T_b, G, leaves_b)

    update_best_graph(T_k, id, 'mst')
    update_best_graph(T_p, id, 'mst')
    update_best_graph(T_b, id, 'mst')

def mds(G, id):
    min_set = approximation.dominating_set.min_weighted_dominating_set(G.copy())
    steiner_tree = approximation.steinertree.steiner_tree(G.copy(), min_set)
    if steiner_tree.number_of_nodes() == 0 and steiner_tree.size() == 0: # steiner tree outputted empty graph
        graph = nx.Graph()
        graph.add_node(min_set.pop())
        update_best_graph(graph, id, 'mds complete')
    else:
        #print(steiner_tree.nodes)
        
        update_best_graph(steiner_tree, id, 'mds')

def random_mds(G, id):
    def span(node, white_set):
        counter = 1
        for neighbor in G.neighbors(node):
            if neighbor in white_set:
                counter += 1
        return counter

    nodes = list(G.copy().nodes())
    random.shuffle(nodes)
    #The priority for node i is denoted by rand_tiebreakers[i]
    rand_tiebreakers = nodes.copy()
    random.shuffle(rand_tiebreakers)
    white = set(nodes)
    black = list()
    gray = set()
    if (len(nodes) == 1):
        black = nodes
    else:
        while len(white) != 0:
            max_node = next(iter(white))
            max_span = span(max_node, white)
            for node in white:
                node_span = span(node, white)
                #print(rand_tiebreakers[node] > rand_tiebreakers[max_node])
                #print(rand_tiebreakers[node], rand_tiebreakers[max_node])
                if node_span > max_span or (node_span == max_span and rand_tiebreakers[node] > rand_tiebreakers[max_node]):
                    max_span = node_span
                    max_node = node
            white.remove(max_node)
            for neighbor in G.neighbors(max_node):
                if neighbor in white:
                    gray.add(neighbor)
                    white.remove(neighbor)
            black.append(max_node)

    #print(sorted(black))
    steiner_tree = approximation.steinertree.steiner_tree(G, black)

    # steiner_tree = steiner_tree.copy()
    # for node in steiner_tree:
    #     neighbors = list(steiner_tree.neighbors(node))
    #    # print(neighbors)
    #     if len(neighbors) == 1:
    #         v = neighbors[0]
    #         weight = G.get_edge_data(node, v)['weight']
    #         steiner_tree.remove_edge(node, v)
    #         if not is_valid_network(steiner_tree, G):
    #             steiner_tree.add_edge(node, v, weight=weight)
    #         else:
    #             print('good')

    score = average_pairwise_distance_fast(steiner_tree)

    T_copy = steiner_tree.copy()
    #print(list(T_copy))
    for v in list(T_copy):
        for w in G.neighbors(v):
            if w not in T_copy:
                T_copy.add_edge(v, w, weight=G.get_edge_data(v, w)['weight'])
                new_score = average_pairwise_distance_fast(T_copy)
                if new_score > score:
                    T_copy.remove_edge(v, w)
                    T_copy.remove_node(w)
                else:
                    score = new_score
                    #print('good')

    #print(score - best_scores[id])
    steiner_tree = T_copy

    if steiner_tree.number_of_nodes() == 0 and steiner_tree.size() == 0: # steiner tree outputted empty graph
        graph = nx.Graph()
        print(id)
        graph.add_node(black.pop())
        update_best_graph(graph, id, 'random mds complete')
    else:
        update_best_graph(steiner_tree, id, 'random mds')


    

def bfs(G, id):
    tree = nx.Graph()
    nodes = list(G.nodes)
    tree.add_nodes_from(nodes)
    visited = {i: False for i in nodes}
    source = np.random.choice(nodes, 1)[0]
    maximum = max_weight[id]

    queue = deque()
    leaves = deque()

    # Mark the source node as visited and enqueue it 
    queue.append(source) 
    visited[source] = True

    while queue: 
        # Dequeue a vertex from queue and print it 
        s = queue.popleft() 

        # Get all adjacent vertices of the dequeued vertex s. If a adjacent has not been visited, then mark it visited and enqueue it 
        neighbors = list(G.neighbors(s))
        n = np.random.permutation(neighbors)
        children = 0
        for i in n: 
            if visited[i] == False: 
                tree.add_edge(s, i, weight = G[s][i]['weight'])
                queue.append(i) 
                visited[i] = True
                children += 1
        if children == 0:
            leaves.append(s)
    if tree.number_of_nodes() == len(leaves) or tree.number_of_nodes() - 1 == len(leaves):
        tree = nx.Graph()
        tree.add_node(source)
    else:
        score = average_pairwise_distance_fast(tree)

        while leaves:
            # p = 1 - ((tree[parent][l]['weight'])/maximum)
            # r = np.random.random()
            # if r < p:
            #     tree.remove_node(l)
            l = leaves.popleft()
            parent = list(tree.neighbors(l))[0]
            edge_weight = tree.get_edge_data(parent, l)['weight']
            tree.remove_node(l)
            if nx.is_dominating_set(G, tree.nodes):
                new_score = average_pairwise_distance_fast(tree)
                p = np.random.random()
                if new_score > score and p > 0.6:
                    tree.add_node(l)
                    tree.add_edge(l, parent, weight=edge_weight)
                else:
                    score = new_score
                    #append new leaves
                    if tree.degree(parent) == 1:
                        leaves.append(parent)
            else:
                tree.add_node(l)
                tree.add_edge(l, parent, weight=edge_weight)

    #print(best_scores[id])
    update_best_graph(tree, id, 'bfs')

# replaces the best graph if the current graph is better
def update_best_graph(G, id, method):
    def write_best_graph():
        filename = id.split('.')[0]
        write_output_file(G, OUTPUT_PATH + filename + '.out')
        print(method + ' ' + id)

    if not is_valid_network(inputs[id], G):
        print("ERROR: " + id + ' ' + method + ' ')
        #print(nx.is_tree(inputs[id]))
        
    if G.number_of_nodes() == 1: # put this in finished files
        if id not in finished_files:
            write_best_graph()
            best_scores[id] = 0
            finished_files.add(id)
            best_methods[id] = method
    else:
        current_score = average_pairwise_distance_fast(G)
        if current_score < best_scores[id] and is_valid_network(inputs[id], G):
            write_best_graph()
            best_scores[id] = current_score
            best_methods[id] = method
    
def write_best_methods():
    methods_file = open(METHODS_PATH, 'w+')
    for id in best_methods:
        methods_file.write(id + ': ' + best_methods[id] + ' ' + str(best_scores[id]) + '\n')

def write_finished_files():
    f = open(FINISHED_FILE_PATH, 'w+')
    for id in finished_files:
        f.write(id + '\n')
    f.close()


# Have tree T- compute cost
# Find all leaf nodes (if we pop a vertex, if no new vertices are added to the queue, it is a leaf)
# Iterate through them, for each leaf, remove on with probability p
# Nw have T*
# Check cost see if its better --> change accordingly

# Repeat for all graphs

#STORAGE
# store each graph somehow
    #for all e in v --> accessing one v and outputting list of edges
# store each tree somehow
    # if we come up with a better tree, write it to output file







# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

if __name__ == '__main__':
    # assert len(sys.argv) == 2
    # path = sys.argv[1]
    # G = read_input_file(path)
    # T = solve(G)
    # assert is_valid_network(G, T)
    # print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
    # write_output_file(T, 'out/test.out')
    solve()
