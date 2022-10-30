<p align="center">
<img src="braingb_logo.png" width="100%" class="center" alt="logo"/>
</p>

<!-- BrainGB is an open-source benchmark package for Brain Network Analysis with Graph Neural Networks based on [PyTorch](https://pytorch.org) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/). It features modularized design space of interest of GNNs for brain networks, and standardized evaluation. -->
BrainGB is a *unified*, *modular*, *scalable*, and *reproducible* framework established for brain network analysis with GNNs. It is designed to enable fair evaluation with accessible datasets, standard settings, and baselines to foster a collaborative environment within computational neuroscience and other related communities. This library is built upon [PyTorch](https://pytorch.org) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/).

To foster research, we provide an out-of-box package that can be installed directly using pip, with detailed tutorials available on our hosted [website](https://brainnet.us/). For more details, please check our paper [here](https://hejiecui.com/files/papers/braingb.pdf).

---
[![HennyJie - BrainGB](https://img.shields.io/static/v1?label=HennyJie&message=BrainGB&color=blue&logo=github)](https://github.com/HennyJie/BrainGB "Go to GitHub repo")
[![stars - BrainGB](https://img.shields.io/github/stars/HennyJie/BrainGB?style=social)](https://github.com/HennyJie/BrainGB)
[![forks - BrainGB](https://img.shields.io/github/forks/HennyJie/BrainGB?style=social)](https://github.com/HennyJie/BrainGB)
![language](https://img.shields.io/github/languages/top/HennyJie/BrainGB?color=lightgrey)
![lines](https://img.shields.io/tokei/lines/github/HennyJie/BrainGB?color=red)
![license](https://img.shields.io/github/license/HennyJie/BrainGB)
![visitor](https://visitor-badge.glitch.me/badge?page_id=BrainGB)
![issue](https://img.shields.io/github/issues/HennyJie/BrainGB)
---

 

<!-- # BrainGB

BrainGB is a python package for testing Graph Neural Networks on Brain Networks.  -->
# Library Highlights
Our BrainGB implements four main modules of GNN models for brain network analysis:
* **Node feature construction**: studies practical and effective methods to initialize either positional or structural node features for each brain region.
* **Message passing mechanisms**: update the node representation of each brain region iteratively by aggregating neighbor features through local connections.
* **Attention-enhanced message passing**: incorporates attention mechanism to enhance the message passing scheme of GNNs. 
* **Pooling strategies**: operate on the set of node vectors to get a graph-level representation.

BrainGB also implements utility functions for model training, performance evaluation, and experiment management.

# Installation

To install BrainGB as a package, simply run
```shell
pip install BrainGB
```

Alternatively, you can also download the repository from Github. The main package is under the src folder. If you choose to go with this method, please check the [Specification of Dependencies](#Specification-of-Dependencies) section for dependency requirements. 

# Specification of Dependencies

BrainGB depends on the following frameworks:

```
torch~=1.10.2
numpy~=1.22.2
nni~=2.4
PyYAML~=5.4.1
scikit-learn~=1.0.2
networkx~=2.6.2
scipy~=1.7.3
tensorly~=0.6.0
pandas~=1.4.1
libsvm~=3.23.0.4
matplotlib~=3.4.3
tqdm~=4.62.3
torch-geometric~=2.0.3
h5py~=3.6.0
```

To install the dependencies, run:
```shell
pip install -r requirements.txt
```

Notice that if you install the package through pip, the dependencies are automatically installed. 


# Getting Started

To import the models detailed in the paper:
```Python
from BrainGB.models import GAT, GCN, BrainNN, GCN
```

The BrainNN is required and will be served as the parent module of the GAT, GCN models. You may choose either GAT or GCN as the submodule. 

To initialize a GCN model
```Python
sample: Data = Data()  # A torch geometric data

num_features = data.x.shape[1]
num_nodes = data.x.shape[0]
gcn_model = GCN(num_features, num_nodes)

model = BrainNN(args.pooling, gcn_model, MLP(2 * num_nodes))
```

To initialize a GAT model, simply replace the GCN with GAT. Both models are customizable. Please refer to the [Customizing Your Own GNN Models](#Customizing-Your-Own-GNN-Models) section for more details. 


# Customizing Your Own GNN Models

## Node Feature Construction
In `src.dataset.tranforms`, BrainGB provides the `BaseTransform` base class, which offers a universal interface for node feature initialization for each brain region. Specifically, BrainGB implements the following node feature construction functions: 

| Node Features                            | Option Name        |
| --------------------------------------- | ----------------- |
| Identity                        | `identity`      |
| Eigen                      | `eigenvector`    |
| Degree                   | `degree`  |
| Degree Profile                    | `LDP`  |
| Connection Profile           | `adj` |

To adjust the type of node features, simply set the chosen option name for the input parameter `node_features`.

## Message Passing Mechanisms
In `models.gcn`, BrainGB provides the base class `MPGCNConv` and different message vector designs including: 
| Message Passing Mechanisms                    | Option Name          |
| ------------------------------------ | ------------------- |
| Edge Weighted | `weighted_sum`  |
| Bin Concat        | `bin_concate` |
| Edge Weight Concat | `edge_weight_concate`  |
| Node Edge Concat        | `edge_node_concate` |
| Node Concat        | `node_concate` |

To adjust the message passing schemes, simply set the input parameter `model_name` as `gcn` and chose an option name for the parameter `gcn_mp_type`.

## Attention-Enhanced Message Passing
In `models.gat`, BrainGB provides the base class `MPGATConv` and different versions of attention-enhanced message passing designs including:
| Message Passing Mechanisms                    | Option Name          |
| ------------------------------------ | ------------------- |
| Attention Weighted | `attention_weighted`  |
| Edge Weighted w/ Attn        | `attention_edge_weighted` |
| Attention Edge Sum | `sum_attention_edge`  |
| Node Edge Concat w/ Attn        | `edge_node_concate` |
| Node Concat w/ Attn        | `node_concate` |

Note that some of these options are corresponding attention enhanced version of the message passing mechanism designs. Please refer to our paper for more details.

To adjust the attention-enhanced message passing schemes, simply set the input parameter `model_name` as `gat` and chose an option name for the parameter `gat_mp_type`.

## Pooling Strategies
The pooling strategy is controlled by setting the `self.pooling` in the chosen model. Specifically, BrainGB implements the following three basic pooling strategies: 
| Pooling Strategies                    | Option Name          |
| ------------------------------------ | ------------------- |
| Mean Pooling | `mean`  |
| Sum Pooling        | `sum` |
| Concat Pooling | `concat`  |

To adjust the pooling strategies, simply set the chosen option name for the input parameter `pooling`.

## 

# Running Example Scripts

The repository also comes with example scripts. To train our model on any of the datasets we tested, simply run:
```shell
python -m main.example_main --dataset_name=<dataset_name> [--model_name=<model_name> --gcn_mp_type=<mp_mechanism>  --gat_mp_type=<attention_mp_mechanism> --node_features=<feature_name> --pooling=<pooling_name> --n_GNN_layer=<GNN_num> --n_MLP_layers=<MLP_num> --hidden_dim=<hidden_layer_dimension> --epochs=<epoch_num> --k_fold_splits=<split_num> --test_interval=<evaluation_interval_num>]
```

The `dataset_name` is the name of the dataset to use (required parameter). We include the following four datasets in our paper:

- HIV
- PNC (Can be downloaded [here](https://www.nitrc.org/projects/pnc/))
- PPMI (Can be downloaded [here](https://www.ppmi-info.org/access-data-specimens/download-data))
- ABCD (Can be downloaded [here](https://nda.nih.gov/abcd))

You can also construct your own datasets by following the [instructions](https://brainnet.us/instructions/) on neuroimaging preprocessing and brain network construction on our website.

Please place the dataset files in the `datasets` folder under the package examples folder. Create the folder if it does not exist.

The `model_name` specifies the backbone model type. Choose `gcn` to test the message passing variants without attention and `gat` to test the attention-enhanced message passing mechanisms. Specifically, use `gcn_mp_type` to set a message vector design and use `gat_mp_type` to set an attention-enhancing mechanism.

The `node_features` specifies the artificial node feature initialization for each brain region.

The `pooling` specifies the pooling strategy to get a graph-level representation for each subject.

You can also change other hyper-parameters, such as `--n_GNN_layer`, `--n_MLP_layers`, `--hidden_dim`, `--epochs`, etc., to adjust the detailed model design or control the training process. All those hyper-parameters can be automatically searched and optimized using the AutoML tool [NNI](https://github.com/microsoft/nni) by passing `--enable_nni`.


# Contribution

Feel free to open an [issue](issues/new) should you find anything unexpected or [create pull requests](pulls) to add your own work! We welcome contributions to this benchmark work and the package.

# Citation

Please cite our paper if you find this code useful for your work:

```
@article{cui2022braingb,
author = {Cui, Hejie and Dai, Wei and Zhu, Yanqiao and Kan, Xuan and Chen Gu, Antonio Aodong and Lukemire, Joshua and Zhan, Liang and He, Lifang and Guo, Ying and Yang, Carl},
title = {{BrainGB: A Benchmark for Brain Network Analysis with Graph Neural Networks}},
journal={ArXiv.org},
year = {2022},
}
```
