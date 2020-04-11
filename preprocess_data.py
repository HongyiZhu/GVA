import os

node_file = "./data/repo_nodes.csv"
features = "./data/repo_feature_matrix_reduced_for_graph.csv"
edge_file = "./data/repo_graph_projection_edge_list.txt"
dataset_name = "repo"
output_features = "./data/{}.features".format(dataset_name)
output_edgelist = "./data/{}.edgelist".format(dataset_name)

def main():
    os.rename(edge_file, output_edgelist)

    nodes = {}
    # read nodes
    f = open(node_file, 'r')
    for i, l in enumerate(f.readlines()):
        # repoID => nodeID mapping
        nodes[l.strip()] = str(i)
    f.close()

    # read lines
    f = open(features, 'r')
    g = open(output_features, 'w')
    f.readline()
    for l in f.readlines():
        # convert repoID to nodeID
        vec = l.strip().split(",")
        g.write("{} ".format(nodes[vec[0]]))
        g.write(" ".join(vec[1:]))
        g.write("\n")
    g.close()
    f.close()


if __name__ == "__main__":
    main()
