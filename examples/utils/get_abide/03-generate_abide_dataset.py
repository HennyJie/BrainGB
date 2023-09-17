import deepdish as dd
import os.path as osp
import os
import numpy as np
import argparse
from pathlib import Path
import pandas as pd


def main(args):
    data_dir =  os.path.join(args.root_path, 'ABIDE_pcp/cpac/filt_noglobal/raw')
    timeseires = os.path.join(args.root_path, 'ABIDE_pcp/cpac/filt_noglobal/')

    meta_file = os.path.join(args.root_path, 'ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv')

    meta_file = pd.read_csv(meta_file, header=0)

    id2site = meta_file[["subject", "SITE_ID"]]

    # pandas to map
    id2site = id2site.set_index("subject")
    id2site = id2site.to_dict()['SITE_ID']

    times = []

    labels = []
    pcorrs = []

    corrs = []

    site_list = []

    for f in os.listdir(data_dir):
        if osp.isfile(osp.join(data_dir, f)):
            fname = f.split('.')[0]
            site = id2site[int(fname)]
            

            files = os.listdir(osp.join(timeseires, fname))

            file = list(filter(lambda x: x.endswith("1D"), files))[0]

            time = np.loadtxt(osp.join(timeseires, fname, file), skiprows=0).T

            if time.shape[1] < 100:
                continue

            temp = dd.io.load(osp.join(data_dir,  f))
            pcorr = temp['pcorr'][()]

            pcorr[pcorr == float('inf')] = 0

            att = temp['corr'][()]

            att[att == float('inf')] = 0

            label = temp['label']

            times.append(time[:,:100])
            labels.append(label[0])
            corrs.append(att)
            pcorrs.append(pcorr)
            site_list.append(site)

    np.save(Path(args.root_path)/'ABIDE_pcp/abide.npy', {'timeseires': np.array(times), "label": np.array(labels),"corr": np.array(corrs),"pcorr": np.array(pcorrs), 'site': np.array(site_list)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the final dataset')
    parser.add_argument('--root_path', default="", type=str, help='The path of the folder containing the dataset folder.')
    args = parser.parse_args()
    main(args)
