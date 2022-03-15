# BrainGB

BrainGB is a python package for testing Graph Neural Networks on Brain Networks. 

# Installation

To install BrainGB as a package, simply run
```shell
pip install BrainGB
```

Alternatively, you can also download the repository from Github. The main package is under the src folder. If you choose to go with this method, please check the [Specification of Dependencies](#Specification-of-Dependencies) section for dependency requirements. 

# Getting Started

To import the models detailed in the paper:
```pycon
from BrainGB import GAT, GCN, BrainNN, GCN
```

The BrainNN is required and will be served as the parent module of the GAT, GCN models. You may choose either GAT or GCN as the submodule. 

To initialize a GCN model
```pycon
sample: Data = Data()  # A torch geometric data

num_features = data.x.shape[1]
num_nodes = data.x.shape[0]
gcn_model = GCN(num_features, num_nodes)

model = BrainNN(args.pooling, gcn_model, MLP(2 * num_nodes))
```

To initialize a GAT model, simply replace the GCN with GAT. Both models are customizable. Please refer to the documentation for more details. 

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

# Running Example Scripts

The repository also comes with example scripts. To train our model on any of the datasets we tested, simply run:
```shell
python -m main.example_main --dataset_name=<dataset_name> [--model_name=<variant>]
```

The `dataset_name` is the name of the dataset you want to use. The following datasets are available and tested:

- HIV
- PPMI (Can be downloaded [here](https://www.ppmi-info.org/access-data-specimens/download-data))
- PNC

The following datasets are also available but not tested:

- ABIDE
- BP
- ABCD

Please place the dataset files in the `datasets` folder under the package examples folder. Create the folder if it does not exist.

The `model_name` is the name of the model you want to use. Two choices are available:

- GCN
- GAT

# Contribution

We welcome contributions to the package. Please feel free to open an issue or pull request. 