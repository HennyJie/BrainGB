#Written by Tony Chen Gu <achengu@emory.edu>
#Visit us at website: https://braingb.us/

import os
from nilearn import connectome
from nibabel import cifti2
import numpy as np
import scipy.io

#Change to the path to the folder where the data is stored
dir_path = os.getcwd()                                   
##########################################################
output_path = os.path.join(dir_path, 'connectivity_output')

# To convert single file, just uncomment the following line, and move the script to the same directory as the file, and enter the full file name in the "file_name"
#files = ["your_full_file_name.ptseries.nii"]

# To convert all files in a directory, uncomment the following line.
files = [f for f in os.listdir(dir_path) if f.endswith('.ptseries.nii')]

###########################################################
#do not change the codes below
if(not os.path.exists(output_path)):
    os.mkdir(output_path)
for file in files:
    file_path = os.path.join(dir_path, file)
    timeseries = np.array([cifti2.cifti2.load(file_path).get_data().tolist()])
    # you could refer to the document of connectome.ConnectivityMeasure to change the way to calculate the connectivity matrix
    # kind{“covariance”, “correlation”, “partial correlation”, “tangent”, “precision”}
    conn_measure = connectome.ConnectivityMeasure(kind='correlation')
    connectivity_fit = conn_measure.fit(timeseries)
    connectivity = connectivity_fit.transform(timeseries)

    new_file_name = os.path.splitext(file)[0]+"_conn.mat"
    scipy.io.savemat(os.path.join(output_path, new_file_name), mdict={'connectivity': connectivity[0]})