import networkx as nx
from networkx.algorithms import approximation
import numpy as np
from parse import read_input_file, write_output_file, read_output_file
from utils import is_valid_network, average_pairwise_distance_fast
import sys
import os
import solver

INPUT_PATH = '../inputs/'
OUTPUT_PATH = '../outputs2/'
FINISHED_FILE_PATH = '../finished_files.txt'
METHODS_PATH = '../methods.txt'
inputs = {}
# dictionary that holds id, score
best_scores = {}
best_methods = {}

finished_files = set()

# def setup():
#     for filename in os.listdir(INPUT_PATH):
#         inputs[filename] = read_input_file(INPUT_PATH + filename)
#         if os.path.isfile(OUTPUT_PATH + filename):
#             output_graph = read_output_file(OUTPUT_PATH + filename, inputs[filename])
#             best_scores[filename] = average_pairwise_distance_fast(output_graph)
#         else:
#             best_scores[filename] = float('inf')

solver.setup(inputs, best_scores, best_methods, finished_files)

def random_mds(G, id):
    dom_set = []
    nodes = list(G.copy().nodes())
    if (len(nodes) == 1):
        dom_set = nodes
    edges = list(G.copy().edges())
    print(nodes)
    while edges:
        # Pick a random edge and random endpoint for that edge to add to dom_set
        edge = edges[np.random.randint(len(edges))]
        vertex = edge[np.random.randint(2)]
        dom_set.append(vertex)
        remove_vertices = set()
        iterated_edges = [edge for edge in edges if (edge[0] == vertex or edge[1] == vertex)]
        for edge in iterated_edges:
            remove_vertices.add(edge[0])
            remove_vertices.add(edge[1])
        edges = [edge for edge in edges if (edge[0] not in remove_vertices and edge[1] not in remove_vertices)]
    print(dom_set)
    steiner_tree = approximation.steinertree.steiner_tree(G, dom_set)
    return steiner_tree

#print(inputs['small-252.in'])
G = inputs['medium-52.in']
for i in range(5):
    T = random_mds(G.copy(), 'medium-52.in')
    print(is_valid_network(G, T))
    # print(average_pairwise_distance_fast(T))
    # print(T.edges())