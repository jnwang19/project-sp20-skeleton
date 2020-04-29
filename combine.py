OUTPUT_PATH = ''

def combine(folders):
    best_scores = {}
    best_outputs = {}

    for folder in folders:
        for filename in os.listdir(folder):
            output_graph = read_output_file(OUTPUT_PATH + out_filename, inputs[filename])
            if output_graph.number_of_nodes() == 1:
                    best_scores[filename] = 0
                    best_outputs[filename] = output_graph
                else:
                    score = average_pairwise_distance_fast(output_graph)
                    if score < best_scores[filename]:
                        best_scores[filename] = score
                        best_outputs[filename] = output_graph

    for id in best_outputs:
        write_output_file(best_outputs[id], OUTPUT_PATH + id)

if __name__ == '__main__':
    folders = ['outputs-jonathan', 'outputs-isabelle', 'outputs-sean']
    combine(folders)