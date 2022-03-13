import numpy as np
import pandas as pd
from sklearn import preprocessing


def load_data_abcd(abcd_path):
    fc_data = np.load(abcd_path + '/abcd_rest-timeseires-HCP2016.npy', allow_pickle=True) # (7907, 360, 512), time_step_num: 512
    pearson_data = np.load(abcd_path + '/abcd_rest-pearson-HCP2016.npy', allow_pickle=True) # (9563, 360, 360), ROI_num: 360
    label_df = pd.read_csv(abcd_path + '/id2sex.txt')  # (9557, 2)

    with open(abcd_path + '/ids_HCP2016.txt', 'r') as f:
        lines = f.readlines()
        pearson_id = [line[:-1] for line in lines] # len(pearson_id): 9563

    with open(abcd_path + '/ids_HCP2016_timeseires.txt', 'r') as f:
        lines = f.readlines()
        fc_id = [line[:-1] for line in lines]  # len(fc_id): 7907

    id2pearson = dict(zip(pearson_id, pearson_data))  # len(id2pearson): 9563

    id2gender = dict(zip(label_df['id'], label_df['sex']))  # len(id2gender): 9557

    final_pearson, labels = process_dataset(fc_data, fc_id, id2gender, id2pearson, label_df)
    # final_fc.shape: torch.Size([7901, 360, 512]),
    # final_pearson.shape: torch.Size([7901, 360, 360]),
    # labels.shape: torch.Size([7901])
    return final_pearson, labels


def process_dataset(fc_data, fc_id, id2gender, id2pearson, label_df):
    final_label, final_pearson = [], []
    for fc, l in zip(fc_data, fc_id):
        if l in id2gender and l in id2pearson:
            if not np.any(np.isnan(id2pearson[l])):
                final_label.append(id2gender[l])
                final_pearson.append(id2pearson[l])
    final_pearson = np.array(final_pearson)  # (7901, 360, 360)
    _, _, node_feature_size = final_pearson.shape  # node_feature_size: 360
    encoder = preprocessing.LabelEncoder()
    encoder.fit(label_df["sex"])
    labels = encoder.transform(final_label)
    return final_pearson, labels


def load_data_pnc(pnc_path):
    fc_data = np.load(pnc_path + '/514_timeseries.npy', allow_pickle=True)  # (7907, 360, 512), time_step_num: 512
    pearson_data = np.load(pnc_path + '/514_pearson.npy', allow_pickle=True)  # (9563, 360, 360), ROI_num: 360
    label_df = pd.read_csv(pnc_path + '/PNC_Gender_Age.csv')  # (9557, 2)

    pearson_data, fc_data = pearson_data.item(), fc_data.item()

    pearson_id = pearson_data['id']
    pearson_data = pearson_data['data']
    id2pearson = dict(zip(pearson_id, pearson_data))

    fc_id = fc_data['id']
    fc_data = fc_data['data']

    id2gender = dict(zip(label_df['SUBJID'], label_df['sex']))

    final_pearson, labels = process_dataset(fc_data, fc_id, id2gender, id2pearson, label_df)

    return final_pearson, labels


def load_data_abide(abide_path):
    data = np.load(abide_path + '/abide.npy', allow_pickle=True).item()
    final_pearson = data["corr"]
    labels = data["label"]

    return final_pearson, labels
