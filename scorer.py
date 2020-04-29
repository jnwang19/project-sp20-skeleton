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

# global variables
# dictionary that holds each id, input networkx.Graph
INPUT_PATH = '../inputs/'
OUTPUT_PATH = '../outputs/'
FINISHED_FILE_PATH = 'finished_files.txt'
METHODS_PATH = '../methods.txt'

def scorer(filename):
    input = INPUT_PATH + filename
    print(input)
    out_filename = filename.split('.')[0] + '.out'
    print(OUTPUT_PATH + out_filename)
    if os.path.isfile(OUTPUT_PATH + out_filename):
        output_graph = read_output_file(OUTPUT_PATH + out_filename, read_input_file(input))
        print(average_pairwise_distance_fast(output_graph))

scorer('small-28.in')
