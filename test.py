import networkx as nx
import numpy as np
from parse import read_input_file, write_output_file, read_output_file
from utils import is_valid_network, average_pairwise_distance_fast
import sys
import os
#import solver

INPUT_PATH = '../inputs/'
OUTPUT_PATH = '../outputs/'
inputs = {}
# dictionary that holds id, score
best_scores = {}

def setup():
    for filename in os.listdir(INPUT_PATH):
        inputs[filename] = read_input_file(INPUT_PATH + filename)
        if os.path.isfile(OUTPUT_PATH + filename):
            output_graph = read_output_file(OUTPUT_PATH + filename, inputs[filename])
            best_scores[filename] = average_pairwise_distance_fast(output_graph)
        else:
            best_scores[filename] = float('inf')

setup()

def random_mds(G, id):
    dom_set = []
    edges = G.edges().copy()
    while edges:
        # Pick a random edge and random endpoint for that edge to add to dom_set
        edge = edges[np.random.randint(len(edges))]
        vertex = edge[np.random.randint(2)]
        print(edge)
        dom_set.append(vertex)
        edges = [edge for edge in edges if (edge[0] != vertex and edge[1] != vertex)]
    steiner_tree = approximation.steinertree.steiner_tree(G, dom_set)
    update_best_graph(steiner_tree, id, 'rand_mds')

#print(inputs['small-252.in'])
G = inputs['small-252.in']
for i in range(5):
    T = random_mds(inputs['small-252.in'], 'small-252.in')
    print(is_valid_network(G, T))
    print(average_pairwise_distance_fast(T))
    print(T.edges())