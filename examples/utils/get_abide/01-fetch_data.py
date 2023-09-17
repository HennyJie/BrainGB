# Copyright (c) 2019 Mwiza Kunda
# Copyright (C) 2017 Sarah Parisot <s.parisot@imperial.ac.uk>, , Sofia Ira Ktena <ira.ktena@imperial.ac.uk>
#
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

'''
This script mainly refers to https://github.com/kundaMwiza/fMRI-site-adaptation/blob/master/fetch_data.py
'''

from nilearn import datasets
import argparse
from preprocess_data import Reader
import os
import shutil
import sys


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
    print(args)

    root_folder = args.root_path
    data_folder = os.path.join(root_folder, 'ABIDE_pcp/cpac/filt_noglobal/')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    pipeline = args.pipeline
    atlas = args.atlas
    download = args.download

    # Files to fetch

    files = ['rois_' + atlas]

    filemapping = {'func_preproc': 'func_preproc.nii.gz',
                   files[0]: files[0] + '.1D'}


    # Download database files
    if download == True:
        abide = datasets.fetch_abide_pcp(data_dir=root_folder, pipeline=pipeline,
                                         band_pass_filtering=True, global_signal_regression=False, derivatives=files,
                                         quality_checked=False)
    reader = Reader(root_folder, args.id_file_path)
    subject_IDs = reader.get_ids() #changed path to data path
    subject_IDs = subject_IDs.tolist()

    # Create a folder for each subject
    for s, fname in zip(subject_IDs, reader.fetch_filenames(subject_IDs, files[0], atlas)):
        subject_folder = os.path.join(data_folder, s)
        if not os.path.exists(subject_folder):
            os.mkdir(subject_folder)

        # Get the base filename for each subject
        base = fname.split(files[0])[0]

        # Move each subject file to the subject folder
        for fl in files:
            if not os.path.exists(os.path.join(subject_folder, base + filemapping[fl])):
                shutil.move(base + filemapping[fl], subject_folder)

    time_series = reader.get_timeseries(subject_IDs, atlas)

    # Compute and save connectivity matrices
    reader.subject_connectivity(time_series, subject_IDs, atlas, 'correlation')
    reader.subject_connectivity(time_series, subject_IDs, atlas, 'partial correlation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download ABIDE data and compute functional connectivity matrices')
    parser.add_argument('--pipeline', default='cpac', type=str,
                        help='Pipeline to preprocess ABIDE data. Available options are ccs, cpac, dparsf and niak.'
                             ' default: cpac.')
    parser.add_argument('--atlas', default='cc200',
                        help='Brain parcellation atlas. Options: ho, cc200 and cc400, default: cc200.')
    parser.add_argument('--download', default=True, type=str2bool,
                        help='Dowload data or just compute functional connectivity. default: True')
    parser.add_argument('--root_path', default="/home/root/", type=str, help='The path of the folder containing the dataset folder.')
    parser.add_argument('--id_file_path', default="subject_IDs.txt", type=str, help='The path to subject_IDs.txt.')
    args = parser.parse_args()
    main(args)