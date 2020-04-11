# GitHub Vulnerability Analysis (GVA)

## Introduction

This is the code repository for the GitHub Vulnerability Analysis project.

## Dependency Requirement

+ TensorFlow (<=1.14.0)
+ PyTorch
+ NetworkX (>=2.0)
+ GenSim
+ [OpenNE: An Open-Source Package for Network Embedding](https://github.com/thunlp/OpenNE)
+ [GEM: Graph Embedding Methods](https://github.com/palash1992/GEM)

## Dataset

### Preprocessing

The `preprocess_data.py` script parse raw CSVs (node features and nodeID mapping) and rename edge list file as the following standardized format. Please change the file paths in the script before execution.

### Edge Lists

Place edge list files under the `/data` folder with the name of `'<dataset>.edgelist'`. An un-directional edge between nodes `a` and `b` can be denoted with `a<space>b` or `b<space>a`. Each edge takes a new line. For example:

```text
'test.edgelist'
0 1  
1 2  
3 1  
. .
. .
```

### Nodal Features

Nodal features are stored under the `/data` folder titled `'<dataset>.features'`. For `d`-dimension nodal features, each row has `d+1` values, with the index of node v<sub>i</sub> followed by `d` features. For example:

```text
'test.features'
0 0.25 0.35 0.41 ...
1 0.18 0.36 0.24 ...
. ...
. ...
```

## Configurations

All experiment configurations are specified in `graph_embedding_config.py`. Specifically, please specify the name of the experiment (`<experiment_name>`) and the dataset to be experimented (`<dataset>`) each time. Benchmark methods can be selected by modifying the `'models'` `list`. For other parameters, please take a look at the file.

## Output

+ Experiment parameter summary: `/data/<experiment_name>_config.json`
+ Embeddings: `/embeddings/<experiment_name>/<model>.nv`  

    ```text
    #nodes #dim
    n0 e01 e02 e03 ... e0n
    n1 e11 e12 e13 ... e1n
    n2 ...
    .  ...
    ```

+ Evaluation results: `/results/<experiment_name>/results.tsv`
+ t-SNE visualization of clustered embeddings: `/results/<experiment_name>/<clustering algorithm>_<graph embedding model>.svg`
+ Runtime data: `/results/<experiment_name>/experiment.cache`

## Example

`python preprocess_data.py`
`python evaluation.py`
`python plot_tsne.py`
