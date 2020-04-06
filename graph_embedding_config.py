# name the experiment
experiment_name = "user_network"

# configurations
dataset = "user" # "repo"
have_features = False

edgelist_filename = "./data/{}.edgelist".format(dataset)
node_index_filename = "./data/{}.index".format(dataset)
features_filename = "./data/{}.features".format(dataset) if have_features else None

EMBEDDING_PATH = './embeddings/{}/'.format(experiment_name)
REPORT_PATH = './results/{}/'.format(experiment_name)

# evaluation models
models = [
    'LE', 'GF', 'LLE', 'HOPE', 'GraRep',    # Matrix Factorization
    'DeepWalk', 'node2vec',                 # Random Walk
    'SDNE', 'VGAE',                         # Deep Representation Learning 'GATE' & 'CANE' to be implemented
    'LINE',                                 # Edge Reconstruction
    'GCAE'                                  # GCAE
    ]
models = [model.upper() for model in models]

embedding_size = 128            # must be smaller than the number of observations in LLE
epochs = 40                     # for LINE
kstep = 4                       # for GraRep
window_size = 10                # for DeepWalk and Node2vec
walk_length = 80                # for DeepWalk and Node2vec
number_walks = 10               # for DeepWalk and Node2vec
workers = 8                     # for DeepWalk and Node2vec
p = 1                           # for Node2vec
q = 1                           # for Node2vec
encoder_list = [512, 128]       # for SDNE

KMEANS_EVAL = True
DBSCAN_EVAL = False
n_clusters = 8                  # for KMeans
eps = 0.01                      # for DBSCAN

# export configuration file
_f = open("./data/{}_config.json".format(experiment_name), "w")
_d = vars()
_f.write({k:_d[k] for k in _d.keys() if not k.startswith("_")}.__repr__())
_f.close()
