import numpy as np

def G(n,p):
    graph = [] 
    # Recall that we describe a graph as a list enumerating all edges. Node names can be numbers.
    
    for i in range(n):
        graph.append((i,i))
    for i in range(n):
        for j in range(i+1,n):
            if np.random.rand() < p:
                graph.append((i, j))
    
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
        for node1,node2 in graph: # iterate over edges
            if (node1 in connected_nodes and node2 not in connected_nodes) or \
                (node1 not in connected_nodes and node2 in connected_nodes):
                connected_nodes.add(node1)
                connected_nodes.add(node2)
                changed_flag = True
    
    return connected_nodes

graph = G(100, 0.046)
while len(find_connected_component(graph, 1)) != 100:
    graph = G(100, 0.046)

f = open("100.in", "w+")
for edge in graph:
    if edge[0] != edge[1]:
        f.write(str(edge[0]) + ' ' + str(edge[1]) + '\n')
#print(graph)