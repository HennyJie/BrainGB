# Copyright (c) 2019 Mwiza Kunda
# Copyright (C) 2017 Sarah Parisot <s.parisot@imperial.ac.uk>, Sofia Ira Ktena <ira.ktena@imperial.ac.uk>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implcd ied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import os
import warnings
import glob
import csv
import numpy as np
import scipy.io as sio
from nilearn import connectome
import pandas as pd
from scipy.spatial import distance
from scipy import signal
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

# Input data variables


class Reader:

    def __init__(self, root_path, id_file_path=None) -> None:

        root_folder = root_path
        self.data_folder = os.path.join(root_folder, 'ABIDE_pcp/cpac/filt_noglobal')
        self.phenotype = os.path.join(root_folder, 'ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv')
        self.id_file = id_file_path


    def fetch_filenames(self, subject_IDs, file_type, atlas):
        """
            subject_list : list of short subject IDs in string format
            file_type    : must be one of the available file types
            filemapping  : resulting file name format
        returns:
            filenames    : list of filetypes (same length as subject_list)
        """

        filemapping = {'func_preproc': '_func_preproc.nii.gz',
                    'rois_' + atlas: '_rois_' + atlas + '.1D'}
        # The list to be filled
        filenames = []

        # Fill list with requested file paths
        for i in range(len(subject_IDs)):
            os.chdir(self.data_folder)
            find_files = glob.glob('*' + subject_IDs[i] + filemapping[file_type])
            if len(find_files) > 0:
                filenames.append(find_files[0])
            else:
                if os.path.isdir(self.data_folder + '/' + subject_IDs[i]):
                    os.chdir(self.data_folder + '/' + subject_IDs[i])
                    filenames.append(glob.glob('*' + subject_IDs[i] + filemapping[file_type])[0])
                else:
                    filenames.append('N/A')
        return filenames


    # Get timeseries arrays for list of subjects
    def get_timeseries(self, subject_list, atlas_name, silence=False):
        """
            subject_list : list of short subject IDs in string format
            atlas_name   : the atlas based on which the timeseries are generated e.g. aal, cc200
        returns:
            time_series  : list of timeseries arrays, each of shape (timepoints x regions)
        """

        timeseries = []
        for i in range(len(subject_list)):
            subject_folder = os.path.join(self.data_folder, subject_list[i])
            ro_file = [f for f in os.listdir(subject_folder) if f.endswith('_rois_' + atlas_name + '.1D')]
            fl = os.path.join(subject_folder, ro_file[0])
            if silence != True:
                print("Reading timeseries file %s" % fl)
            timeseries.append(np.loadtxt(fl, skiprows=0))

        return timeseries


    #  compute connectivity matrices
    def subject_connectivity(self, timeseries, subjects, atlas_name, kind, iter_no='', seed=1234,
                            n_subjects='', save=True, save_path=None):
        """
            timeseries   : timeseries table for subject (timepoints x regions)
            subjects     : subject IDs
            atlas_name   : name of the parcellation atlas used
            kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
            iter_no      : tangent connectivity iteration number for cross validation evaluation
            save         : save the connectivity matrix to a file
            save_path    : specify path to save the matrix if different from subject folder
        returns:
            connectivity : connectivity matrix (regions x regions)
        """

        if kind in ['TPE', 'TE', 'correlation','partial correlation']:
            if kind not in ['TPE', 'TE']:
                conn_measure = connectome.ConnectivityMeasure(kind=kind)
                connectivity = conn_measure.fit_transform(timeseries)
            else:
                if kind == 'TPE':
                    conn_measure = connectome.ConnectivityMeasure(kind='correlation')
                    conn_mat = conn_measure.fit_transform(timeseries)
                    conn_measure = connectome.ConnectivityMeasure(kind='tangent')
                    connectivity_fit = conn_measure.fit(conn_mat)
                    connectivity = connectivity_fit.transform(conn_mat)
                else:
                    conn_measure = connectome.ConnectivityMeasure(kind='tangent')
                    connectivity_fit = conn_measure.fit(timeseries)
                    connectivity = connectivity_fit.transform(timeseries)

        if save:
            if not save_path:
                save_path = self.data_folder
            if kind not in ['TPE', 'TE']:
                for i, subj_id in enumerate(subjects):
                    subject_file = os.path.join(save_path, subj_id,
                                                subj_id + '_' + atlas_name + '_' + kind.replace(' ', '_') + '.mat')
                    sio.savemat(subject_file, {'connectivity': connectivity[i]})
                return connectivity
            else:
                for i, subj_id in enumerate(subjects):
                    subject_file = os.path.join(save_path, subj_id,
                                                subj_id + '_' + atlas_name + '_' + kind.replace(' ', '_') + '_' + str(
                                                    iter_no) + '_' + str(seed) + '_' + validation_ext + str(
                                                    n_subjects) + '.mat')
                    sio.savemat(subject_file, {'connectivity': connectivity[i]})
                return connectivity_fit


    # Get the list of subject IDs

    def get_ids(self, num_subjects=None):
        """
        return:
            subject_IDs    : list of all subject IDs
        """

        subject_IDs = np.genfromtxt(self.id_file, dtype=str)

        if num_subjects is not None:
            subject_IDs = subject_IDs[:num_subjects]

        return subject_IDs


    # Get phenotype values for a list of subjects
    def get_subject_score(self, subject_list, score):
        scores_dict = {}

        with open(self.phenotype) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if row['SUB_ID'] in subject_list:
                    if score == 'HANDEDNESS_CATEGORY':
                        if (row[score].strip() == '-9999') or (row[score].strip() == ''):
                            scores_dict[row['SUB_ID']] = 'R'
                        elif row[score] == 'Mixed':
                            scores_dict[row['SUB_ID']] = 'Ambi'
                        elif row[score] == 'L->R':
                            scores_dict[row['SUB_ID']] = 'Ambi'
                        else:
                            scores_dict[row['SUB_ID']] = row[score]
                    elif (score == 'FIQ' or score == 'PIQ' or score == 'VIQ'):
                        if (row[score].strip() == '-9999') or (row[score].strip() == ''):
                            scores_dict[row['SUB_ID']] = 100
                        else:
                            scores_dict[row['SUB_ID']] = float(row[score])

                    else:
                        scores_dict[row['SUB_ID']] = row[score]

        return scores_dict


    # preprocess phenotypes. Categorical -> ordinal representation
    @staticmethod
    def preprocess_phenotypes(pheno_ft, params):
        if params['model'] == 'MIDA':
            ct = ColumnTransformer([("ordinal", OrdinalEncoder(), [0, 1, 2])], remainder='passthrough')
        else:
            ct = ColumnTransformer([("ordinal", OrdinalEncoder(), [0, 1, 2, 3])], remainder='passthrough')

        pheno_ft = ct.fit_transform(pheno_ft)
        pheno_ft = pheno_ft.astype('float32')

        return (pheno_ft)


    # create phenotype feature vector to concatenate with fmri feature vectors
    @staticmethod
    def phenotype_ft_vector(pheno_ft, num_subjects, params):
        gender = pheno_ft[:, 0]
        if params['model'] == 'MIDA':
            eye = pheno_ft[:, 0]
            hand = pheno_ft[:, 2]
            age = pheno_ft[:, 3]
            fiq = pheno_ft[:, 4]
        else:
            eye = pheno_ft[:, 2]
            hand = pheno_ft[:, 3]
            age = pheno_ft[:, 4]
            fiq = pheno_ft[:, 5]

        phenotype_ft = np.zeros((num_subjects, 4))
        phenotype_ft_eye = np.zeros((num_subjects, 2))
        phenotype_ft_hand = np.zeros((num_subjects, 3))

        for i in range(num_subjects):
            phenotype_ft[i, int(gender[i])] = 1
            phenotype_ft[i, -2] = age[i]
            phenotype_ft[i, -1] = fiq[i]
            phenotype_ft_eye[i, int(eye[i])] = 1
            phenotype_ft_hand[i, int(hand[i])] = 1

        if params['model'] == 'MIDA':
            phenotype_ft = np.concatenate([phenotype_ft, phenotype_ft_hand], axis=1)
        else:
            phenotype_ft = np.concatenate([phenotype_ft, phenotype_ft_hand, phenotype_ft_eye], axis=1)

        return phenotype_ft


    # Load precomputed fMRI connectivity networks
    def get_networks(self, subject_list, kind, iter_no='', seed=1234, n_subjects='', atlas_name="aal",
                    variable='connectivity'):
        """
            subject_list : list of subject IDs
            kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
            atlas_name   : name of the parcellation atlas used
            variable     : variable name in the .mat file that has been used to save the precomputed networks
        return:
            matrix      : feature matrix of connectivity networks (num_subjects x network_size)
        """

        all_networks = []
        for subject in subject_list:
            if len(kind.split()) == 2:
                kind = '_'.join(kind.split())
            fl = os.path.join(self.data_folder, subject,
                                subject + "_" + atlas_name + "_" + kind.replace(' ', '_') + ".mat")


            matrix = sio.loadmat(fl)[variable]
            all_networks.append(matrix)

        if kind in ['TE', 'TPE']:
            norm_networks = [mat for mat in all_networks]
        else:
            norm_networks = [np.arctanh(mat) for mat in all_networks]

        networks = np.stack(norm_networks)

        return networks

