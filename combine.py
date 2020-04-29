from parse import read_input_file, write_output_file, read_output_file
from utils import is_valid_network, average_pairwise_distance_fast
import os

INPUT_PATH = '../inputs/'
OUTPUT_PATH = '../outputs-combined/'

def combine(folders):
    inputs = {}
    best_scores = {}
    best_outputs = {}

    for filename in os.listdir(INPUT_PATH):
        inputs[filename] = read_input_file(INPUT_PATH + filename)

    for folder in folders:
        for filename in os.listdir(folder):
            if filename != '.DS_Store':
                output_graph = read_output_file(folder + filename, inputs[filename.split('.')[0] + '.in'])
                if output_graph.number_of_nodes() == 1:
                    best_scores[filename] = 0
                    best_outputs[filename] = output_graph
                else:
                    score = average_pairwise_distance_fast(output_graph)
                    if filename not in best_scores:
                        best_scores[filename] = score
                        best_outputs[filename] = output_graph
                    elif filename in best_scores and score < best_scores[filename]:
                        best_scores[filename] = score
                        best_outputs[filename] = output_graph

    for id in best_outputs:
        write_output_file(best_outputs[id], OUTPUT_PATH + id)

if __name__ == '__main__':
    folders = ['../outputs-mac/', '../outputs-old/']
    combine(folders)