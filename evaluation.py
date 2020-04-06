import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from scipy                  import stats
from sklearn.cluster        import KMeans, DBSCAN
from sklearn                import metrics
from mygraph                import Graph_Int, Graph_Str
from build_embedding        import *
from gem.evaluation         import evaluate_graph_reconstruction as gr
from graph_embedding_config import *
import numpy as np
import networkx as nx
import pickle
import time
import json


# evaluate embedding
def evaluate_embedding(graph, embedding):   
    class _Dummy(object):
        def __init__(self, embedding):
            self.embedding = embedding

        def get_reconstructed_adj(self, X=None, node_l=None):
            node_num = self.embedding.shape[0]
            adj_mtx_r = np.zeros((node_num, node_num))
            for v_i in range(node_num):
                for v_j in range(node_num):
                    if v_i == v_j:
                        continue
                    adj_mtx_r[v_i, v_j] = self.get_edge_weight(v_i, v_j)
            return adj_mtx_r
        
        def get_edge_weight(self, i, j):
            return np.dot(self.embedding[i, :], self.embedding[j, :])

    dummy_model = _Dummy(embedding)
    MAP, prec_curv, err, err_baseline = gr.evaluateStaticGraphReconstruction(graph, dummy_model, embedding, None)
    return (MAP, prec_curv)
    

# evaluate clustering
def evaluate_clustering_performance(X, prediction):
    silhouette = metrics.silhouette_score(X, prediction)
    chs = metrics.calinski_harabasz_score(X, prediction)
    dbs = metrics.davies_bouldin_score(X, prediction)
    
    return (silhouette, chs, dbs)


def process_node_index(edgelist_filename, node_index_filename):
    f = open(edgelist_filename, 'r')
    nodes = []
    for line in f.readlines():
        elements = line.strip().split()
        if len(elements) < 2:
            continue
        else:
            nodes.append(int(elements[0]))
            nodes.append(int(elements[1]))
    f.close()
    nodes = sorted(list(set(nodes)), key=lambda x: int(x))
    nodes_index = {x:i for i, x in enumerate(nodes)}
    f = open(node_index_filename, 'wb')
    pickle.dump(nodes_index, f)
    f.close()


def main():
    process_node_index(edgelist_filename, node_index_filename)
    temp = open(node_index_filename, 'rb')
    node_index = pickle.load(temp)
    temp.close()

    # load dataset
    print("====================\nLoading edgelist")
    t1 = time.time()
    # load graph from edgelist and feature file
    graph = Graph_Int()
    graph.read_edgelist(filename=edgelist_filename, node_index=node_index, weighted=False, directed=False)
    graph_str = Graph_Str()
    graph_str.read_edgelist(filename=edgelist_filename, node_index=node_index, weighted=False, directed=False)
    if have_features:
        graph.read_node_features(filename=features_filename)
    print("Data Loaded. Time elapsed: {:.3f}\n====================\n".format(time.time() - t1))

    # build graph embedding
    print("====================\nBuilding Graph Embeddings\n")
    if not os.path.exists(EMBEDDING_PATH):
        os.makedirs(EMBEDDING_PATH)
    t2 = time.time()
    graph_embeddings = {}
    for model in models:
        graph_embeddings[model] = build_embedding(graph, graph_str, model, EMBEDDING_PATH)
    print("Embeddings Constructed. Total time elapsed: {:.3f}\n====================".format(time.time() - t2))

    # GEM graph reconstruction evaluation
    print("====================\nEvaluating Graph Embeddings")
    t3 = time.time()
    reconstruction_performance = {}
    for model in models:
        reconstruction_performance[model] = evaluate_embedding(graph.G, graph_embeddings[model])
    print("Embeddings Evaluated. Total time elapsed: {:.3f}\n====================".format(time.time() - t3))

    # clustering evaluation
    print("====================\nEvaluating Node Clusters")
    t4 = time.time()
    kmeans_performance = {}
    dbscan_performance = {}
    
    # KMeans
    if KMEANS_EVAL:
        kmeans_prediction = {}
        for model in models:
            print("[KMeans] Clustering {} Embedding".format(model))
            temp_t = time.time()
            kmeans = KMeans(n_clusters=8).fit(graph_embeddings[model])
            kmeans_prediction[model] = kmeans.labels_
            kmeans_performance[model] = evaluate_clustering_performance(graph_embeddings[model], kmeans_prediction[model])
            print("[KMeans] Clustering finished. Time elapsed: {:.3f}".format(time.time() - temp_t))
    
    if DBSCAN_EVAL:
        # DBSCAN
        dbscan_predcition = {}
        for model in models:
            print("[DBSCAN] Clustering {} Embedding".format(model))
            temp_t = time.time()
            dbscan = DBSCAN(eps=0.01).fit(graph_embeddings[model])
            dbscan_predcition[model] = dbscan.labels_
            dbscan_performance[model] = evaluate_clustering_performance(graph_embeddings[model], dbscan_predcition[model])
            print("[DBSCAN] Clustering finished. Time elapsed: {:.3f}".format(time.time() - temp_t))
    print("Clustering Results Evaluated. Total time elapsed: {:.3f}\n====================".format(time.time() - t4))

    # Generate Report
    if not os.path.exists(REPORT_PATH):
        os.makedirs(REPORT_PATH)
    
    f = open("{}results.tsv".format(REPORT_PATH), "w")
    for model in models:
        f.write("{}\t".format(model))
        MAP, prec_curv = reconstruction_performance[model]
        f.write("{:.3f}\t".format(MAP))
        if KMEANS_EVAL:
            k_s, k_c, k_d = kmeans_performance[model]
            f.write("{:.3f}\t{:.3f}\t{:.3f}\t".format(k_s, k_c, k_d))
        if DBSCAN_EVAL:
            d_s, d_c, d_d = dbscan_performance[model]
            f.write("{:.3f}\t{:.3f}\t{:.3f}\t".format(d_s, d_c, d_d))
        f.write("{}\n".format("\t".join(["{:.3f}".format(x) for x in prec_curv[:10]])))
    f.close()
    
    # dump data to cache
    f = open("{}experiment.cache".format(REPORT_PATH), "wb")
    data_cache = [graph_embeddings, reconstruction_performance]
    if KMEANS_EVAL:
        data_cache.append(kmeans_prediction)
        data_cache.append(kmeans_performance)
    if DBSCAN_EVAL:
        data_cache.append(dbscan_predcition)
        data_cache.append(dbscan_performance)
    pickle.dump(data_cache, f)
    f.close()


if __name__ == "__main__":
    main()