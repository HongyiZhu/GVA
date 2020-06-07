# name the experiment
experiment_name = "user_network"

# configurations
dataset = "repo" # "user"
have_features = True
weighted_graph = True

edgelist_filename = "./data/{}.edgelist".format(dataset)
node_index_filename = "./data/{}.index".format(dataset)
embedding_mapping = "./data/{}_mapping.csv".format(dataset)
features_filename = "./data/{}.features".format(dataset) if have_features else None
node_file = "./data/{}_nodes.csv".format(dataset)

EMBEDDING_PATH = './embeddings/{}/'.format(experiment_name)
REPORT_PATH = './results/{}/'.format(experiment_name)

# evaluation models
models = [
    # 'LE', 
    # 'GF', 
    # 'LLE', 
    # 'HOPE', 
    # 'GraRep',    
    # 'DeepWalk', 
    # 'node2vec',                 
    # 'SDNE',                         
    # 'LINE',                                  
    # 'GCAE',
    'TADW',
    'VGAE',
    # 'DANE',
    # 'CANE',
    'GATE',
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
lamb = 0.2                      # for TADW

LOAD_TRAINED_EMBEDDING = False  # Skip embedding construction
KMEANS_EVAL = True              # Use KMeans for clustering
DBSCAN_EVAL = False             # Use DBSCAN for clustering
n_clusters = 8                  # for KMeans
eps = 0.01                      # for DBSCAN

gate_args = {
    'lr': 0.0001,
    'n_epochs': epochs,
    'hidden_dims': [256, embedding_size], 
    'lambda_': 1.0, 
    'dropout': 0.0,
    'gradient_clipping': 5.0,
}

cane_args = {
    'ratio': 0.55,
    'MAX_LEN': 300,
    'neg_table_size': 1000000,
    'NEG_SAMPLE_POWER': 0.75,
    'batch_size': 64,
    'num_epoch': 200,
    'embed_size': 200,
    'lr': 1e-3
}

# export configuration file
_f = open("./data/{}_config.json".format(experiment_name), "w")
_d = vars()
_f.write({k:_d[k] for k in _d.keys() if not k.startswith("_")}.__repr__())
_f.close()
