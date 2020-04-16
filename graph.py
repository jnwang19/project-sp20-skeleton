import numpy as np
import math

def G(n,p):
    graph = [] 
    # Recall that we describe a graph as a list enumerating all edges. Node names can be numbers.
    #bimod = bimodal_distr(2, 5, 1)
    for i in range(n):
        graph.append((i,i, 0))
    for i in range(n):
        for j in range(i+1,n):
            if np.random.rand() < p:
                weight = min(max(round(bimodal_distr(20, 80, 5)), 1), 99)
                graph.append((i, j, weight))
    print(graph)
    return graph


def find_connected_component(graph, starting_node):
    """
    >>> graph = [(1,2),(2,3),(1,3)]
    >>> find_connected_component(graph,1)
    {1, 2, 3}
    >>> graph = [(1,1),(2,3),(2,4),(3,5),(3,6),(4,6),(1,7),(7,8),(1,8)]
    >>> find_connected_component(graph,1)
    {1, 7, 8}
    >>> find_connected_component(graph,2)
    {2, 3, 4, 5, 6}
    """
    connected_nodes = set()
    connected_nodes.add( starting_node )
    
    changed_flag = True
    while changed_flag:
        changed_flag = False
        for node1, node2, weight in graph: # iterate over edges
            if (node1 in connected_nodes and node2 not in connected_nodes) or \
                (node1 not in connected_nodes and node2 in connected_nodes):
                connected_nodes.add(node1)
                connected_nodes.add(node2)
                changed_flag = True
    
    return connected_nodes

def create_file(num, p):
    graph = G(num, p)
    while len(find_connected_component(graph, 1)) != num:
        graph = G(num, p)

    f = open(str(num) + ".in", "w+")
    f.write(str(num) + '\n')
    for edge in graph:
        if edge[0] != edge[1]:
            f.write(str(edge[0]) + ' ' + str(edge[1]) + ' ' + str(edge[2]) + '\n')

def bimodal_distr(mean1, mean2, stdev):
#returns function f --> f() generates a number using the distribution
    x = np.random.normal(mean1, stdev)
    y = np.random.normal(mean2, stdev)
    p = np.random.uniform()
    if p < .5:
        return x
    else:
        return y
    
create_file(50, .15)
create_file(100, 0.65)

p = lambda n, l: l * (np.log(n)/n)
