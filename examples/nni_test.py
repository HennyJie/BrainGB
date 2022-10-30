import random
import argparse
import utils.simple_param as SP
import os


def test_seed(script_name: str, dataset='PPMI', enable_nni=False, num_trials=1000,
              node_features='degree_bin'):
    seeds = []
    num_trials = 10 if enable_nni else num_trials
    print(f'running {num_trials} trials')
    for i in range(num_trials):
        seeds.append(random.randint(100000, 10000000))

    default_param = {
        'dataset_name': dataset,
        'node_features': node_features,
        'weight_decay': 0.0001,
        'initial_epochs': 100,
        'n_MLP_layer': 3,
        'n_GNN_layers': 1,
        'hidden_dim': 16,
        'lr': 0.001,
    }

    sp = SP.SimpleParam(default=default_param)
    params = sp(from_='None', preprocess_nni=False)

    param_str = ' '.join([f'--{k} {v}' for k, v in params.items()])

    cmd = f'python {script_name} {param_str} --enable_val'
    cmd += ' --enable_nni' if enable_nni else ''
    print(cmd)
    os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='example_main.py')
    parser.add_argument('--enable_nni', action='store_true')
    parser.add_argument('--dataset', type=str, default='PPMI')
    parser.add_argument('--trials', type=int, default=1000)
    parser.add_argument('--node_features', type=str,
                        choices=['identity', 'degree', 'degree_bin', 'LDP', 'node2vec', 'adj'],
                        default='adj')
    args = parser.parse_args()

    cwd = os.getcwd()
    print(cwd)

    test_seed(args.target, dataset=args.dataset, enable_nni=args.enable_nni, num_trials=args.trials,
              node_features=args.node_features)
