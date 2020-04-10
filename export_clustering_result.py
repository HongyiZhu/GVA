
from graph_embedding_config         import *
import pickle

# load node_index
# node_index: node_index[nodeID] = EmbeddingID
temp = open(node_index_filename, 'rb')
nodeID2embeddingID = pickle.load(temp)
embeddingID2nodeID = {nodeID2embeddingID[k]: k for k in nodeID2embeddingID.keys()}
temp.close()

# read nodeID & repoID/userID mapping
# nodes: nodes[repoID/userID] = nodeID
origID2nodeID = {}
f = open(node_file, 'r')
for i, l in enumerate(f.readlines()):
    # repoID => nodeID mapping
    origID2nodeID[l.strip()] = str(i)
nodeID2origID = {origID2nodeID[k]: k for k in origID2nodeID.keys()}
f.close()

# load predictions
f = open("{}experiment.cache".format(REPORT_PATH), "rb")
data_cache = pickle.load(f)
if KMEANS_EVAL:
    kmeans_prediction = data_cache[4]
if DBSCAN_EVAL and not KMEANS_EVAL:
    dbscan_prediction = data_cache[4]
elif DBSCAN_EVAL and KMEANS_EVAL:
    dbscan_prediction = data_cache[6]
f.close()

# export as csv
if KMEANS_EVAL and DBSCAN_EVAL:
    f = open("{}clustering_labels.csv".format(REPORT_PATH), 'w')
    f.write("EmbeddingID, NodeID, OriginalID, {}_Kmeans, {}_DBSCAN\n".format(
        "_Kmeans, ".join(models),
        "_DBSCAN, ".join(models)))
    for i in range(len(nodeID2embeddingID.keys())):
        for model in models:
            f.write("{}, {}, {}, {}, {}\n".format(
                str(i), 
                str(embeddingID2nodeID[i]), 
                nodeID2origID[str(embeddingID2nodeID[i])],
                ", ".join([str(kmeans_prediction[model][i]) for model in models]),
                ", ".join([str(dbscan_prediction[model][i]) for model in models])
                ))
    f.close()
elif KMEANS_EVAL:
    f = open("{}clustering_labels.csv".format(REPORT_PATH), 'w')
    f.write("EmbeddingID, NodeID, OriginalID, {}_Kmeans\n".format(
        "_Kmeans, ".join(models))
        )
    for i in range(len(nodeID2embeddingID.keys())):
        for model in models:
            f.write("{}, {}, {}, {}\n".format(
                str(i), 
                str(embeddingID2nodeID[i]), 
                nodeID2origID[str(embeddingID2nodeID[i])],
                ", ".join([str(kmeans_prediction[model][i]) for model in models])
                ))
    f.close()
elif DBSCAN_EVAL:
    f = open("{}clustering_labels.csv".format(REPORT_PATH), 'w')
    f.write("EmbeddingID, NodeID, OriginalID, {}_DBSCAN\n".format(
        "_DBSCAN, ".join(models))
        )
    for i in range(len(nodeID2embeddingID.keys())):
        for model in models:
            f.write("{}, {}, {}, {}\n".format(
                str(i), 
                str(embeddingID2nodeID[i]), 
                nodeID2origID[str(embeddingID2nodeID[i])],
                ", ".join([str(dbscan_prediction[model][i]) for model in models])
                ))
    f.close()


