# Copyright (c) 2019 Mwiza Kunda
# Modified by Xuan Kan
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import sys
import argparse
import pandas as pd
import numpy as np
from preprocess_data import Reader
import deepdish as dd
import warnings
import os

warnings.filterwarnings("ignore")



# Process boolean command line arguments
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(args):

    print('Arguments: \n', args)

    data_folder = os.path.join(args.root_path, 'ABIDE_pcp/cpac/filt_noglobal/')


    params = dict()

    params['seed'] = args.seed  # seed for random initialisation

    # Algorithm choice
    params['atlas'] = args.atlas  # Atlas for network construction
    atlas = args.atlas  # Atlas for network construction (node definition)
    
    reader = Reader(args.root_path, args.id_file_path)
    # Get subject IDs and class labels
    subject_IDs = reader.get_ids()
    labels = reader.get_subject_score(subject_IDs, score='DX_GROUP')

    # Number of subjects and classes for binary classification
    num_classes = args.nclass
    num_subjects = len(subject_IDs)
    params['n_subjects'] = num_subjects

    # Initialise variables for class labels and acquisition sites
    # 1 is autism, 2 is control
    y_data = np.zeros([num_subjects, num_classes]) # n x 2
    y = np.zeros([num_subjects, 1]) # n x 1

    # Get class labels for all subjects
    for i in range(num_subjects):
        y_data[i, int(labels[subject_IDs[i]]) - 1] = 1
        y[i] = int(labels[subject_IDs[i]])

    

    # Compute feature vectors (vectorised connectivity networks)
    fea_corr = reader.get_networks(subject_IDs, iter_no='', kind='correlation', atlas_name=atlas) #(1035, 200, 200)
    fea_pcorr = reader.get_networks(subject_IDs, iter_no='', kind='partial correlation', atlas_name=atlas) #(1035, 200, 200)

    if not os.path.exists(os.path.join(data_folder,'raw')):
        os.makedirs(os.path.join(data_folder,'raw'))
    for i, subject in enumerate(subject_IDs):
        dd.io.save(os.path.join(data_folder,'raw',subject+'.h5'),{'corr':fea_corr[i],'pcorr':fea_pcorr[i],'label':(y[i]-1)})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification of the ABIDE dataset using a Ridge classifier. '
                                                 'MIDA is used to minimize the distribution mismatch between ABIDE sites')
    parser.add_argument('--atlas', default='cc200',
                        help='Atlas for network construction (node definition) options: ho, cc200, cc400, default: cc200.')
    parser.add_argument('--seed', default=123, type=int, help='Seed for random initialisation. default: 1234.')
    parser.add_argument('--nclass', default=2, type=int, help='Number of classes. default:2')
    parser.add_argument('--root_path', default="", type=str, help='The path of the folder containing the dataset folder.')
    parser.add_argument('--id_file_path', default="subject_IDs.txt", type=str, help='The path to subject_IDs.txt.')


    args = parser.parse_args()
    main(args)
