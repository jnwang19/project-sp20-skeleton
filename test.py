import networkx as nx
from networkx.algorithms import approximation
import numpy as np
import random
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

    def span(node, white_set):
        counter = 1
        for neighbor in G.neighbors(n):
            if neighbor in white_set:
                counter += 1
        return counter

    nodes = list(G.copy().nodes())
    #The priority for node i is denoted by rand_tiebreakers[i]
    rand_tiebreakers = random.shuffle(nodes.copy())
    white = set(nodes)
    black = []
    gray = set()
    if (len(nodes) == 1):
        black = nodes
    else:
        while len(white) != 0:
            max_span = -1
            max_node = None
            for node in white:
                node_span = span(node, white)
                if node_span > max_span or (node_span == max_span and rand_tiebreakers[node] > rand_tiebreakers[max_node]):
                    node_span = max_span
                    max_node = node
            white.remove(max_node)
            for neighbor in G.neighbors(max_node):
                if neighbor in white:
                    gray.add(neighbor)
            black.append(neighbor)
    print(black)
    steiner_tree = approximation.steinertree.steiner_tree(G, black)
    return steiner_tree

#print(inputs['small-252.in'])
G = inputs['medium-52.in']
for i in range(5):
    T = random_mds(G.copy(), 'medium-52.in')
    print(is_valid_network(G, T))
    # print(average_pairwise_distance_fast(T))
    # print(T.edges())