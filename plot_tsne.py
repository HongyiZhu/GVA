from time                   import time
from graph_embedding_config import *
import numpy                as np
import matplotlib.pyplot    as plt
import pickle

def plot_embedding(data, label, title, path):
    x_min, x_max = np.min(data), np.max(data)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(i),
                 color=plt.cm.Set1(label[i] / float(np.max(label))),
                 fontdict={'weight': 'regular', 'size': 6})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.savefig(path, format='svg')

def main():
    f = open("{}experiment.cache".format(REPORT_PATH), "rb")
    result = pickle.load(f)
    f.close()
    tsne_result = result[2]
    tsne_time = result[3]
    for model in models:
        if KMEANS_EVAL:
            prediction = result[4]
            path = "{}KMeans_{}.svg".format(REPORT_PATH, model)
            plot_embedding(tsne_result[model], prediction[model],
                         't-SNE visualization of KMeans Clustering of\n{} Embeddings (time {:.3f}s)'.format(model, tsne_time[model]),
                         path)
        if DBSCAN_EVAL and KMEANS_EVAL:
            prediction = result[6]
            path = "{}DBSCAN_{}.svg".format(REPORT_PATH, model)
            plot_embedding(tsne_result[model], prediction[model],
                         't-SNE visualization of DBSCAN Clustering of\n{} Embeddings (time {:.3f}s)'.format(model, tsne_time[model]),
                         path)
        elif DBSCAN_EVAL and not KMEANS_EVAL:
            prediction = result[4]
            path = "{}DBSCAN_{}.svg".format(REPORT_PATH, model)
            plot_embedding(tsne_result[model], prediction[model],
                         't-SNE visualization of DBSCAN Clustering of {} Embeddings (time {:.3f}s)'.format(model, tsne_time[model]),
                         path)

if __name__ == '__main__':
    main()