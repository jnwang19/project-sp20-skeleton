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

# global variables
# dictionary that holds each id, input networkx.Graph
INPUT_PATH = '../inputs/'
OUTPUT_PATH = '../outputs/'
FINISHED_FILE_PATH = 'finished_files.txt'
METHODS_PATH = '../methods.txt'

manager = multiprocessing.Manager()
inputs = manager.dict()
# dictionary that holds id, score
best_scores = manager.dict()
best_methods = manager.dict()
max_weight = manager.dict()
dummy = manager.dict()

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

    for i in range(1):
        print(i)
        processes = []
        for id in inputs:
            # G = inputs[id]
            # random_mds(G.copy(), id)
            # bfs(G.copy(), id)
            p = multiprocessing.Process(target=random_mds, args=[inputs[id].copy(), id])
            p.start()
            processes.append(p)
        for process in processes:
            process.join()
        write_best_methods()
    
    for id in dummy:
        print(id)

    #write_finished_files()
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

def mst(G, id):
    T = nx.minimum_spanning_tree(G)
    num_nodes = T.number_of_nodes()
    leaves = []
    for v in T:
        neighbors = T[v]
        if len(neighbors) == 1 and len(leaves) + 1 < num_nodes:
            leaves.append(v)
    # probabilites
    # if len(leaves) == num_nodes:

    T.remove_nodes_from(leaves)
    update_best_graph(T, id, 'mst')

def mds(G, id):
    min_set = approximation.dominating_set.min_weighted_dominating_set(G)
    steiner_tree = approximation.steinertree.steiner_tree(G, min_set)
    if steiner_tree.number_of_nodes() == 0 and steiner_tree.size() == 0: # steiner tree outputted empty graph
        graph = nx.Graph()
        graph.add_node(min_set.pop())
        update_best_graph(graph, id, 'mds complete')
    else:
        update_best_graph(steiner_tree, id, 'mds')

def random_mds(G, id):
    dummy[id] = id
    def span(node, white_set):
        counter = 1
        for neighbor in G.neighbors(node):
            if neighbor in white_set:
                counter += 1
        return counter

    nodes = list(G.copy().nodes())
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
    steiner_tree = approximation.steinertree.steiner_tree(G, black)
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
    leaves = []

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
        for l in leaves:
            parent = list(tree.neighbors(l))[0]
            p = 1 - ((tree[parent][l]['weight'])/maximum)
            r = np.random.random()
            if r < p:
                tree.remove_node(l)
    update_best_graph(tree, id, 'bfs')

# replaces the best graph if the current graph is better
def update_best_graph(G, id, method):
    def write_best_graph():
        filename = id.split('.')[0]
        write_output_file(G, OUTPUT_PATH + filename + '.out')
        print(method + ' ' + id)

    # if not is_valid_network(inputs[id], G):
    #     print("ERROR: " + id + ' ' + method + ' ')
        
    if G.number_of_nodes() == 1: # put this in finished files
        if id not in finished_files:
            write_best_graph()
            best_scores[id] = 0
            finished_files.add(id)
            best_methods[id] = method
    else:
        current_score = average_pairwise_distance_fast(G)
        if current_score < best_scores[id]:
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
