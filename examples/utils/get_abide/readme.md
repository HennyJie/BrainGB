# Autism Brain Imaging Data Exchange (ABIDE)

The [Autism Brain Imaging Data Exchange (ABIDE)]((http://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html)) collects resting-state functional magnetic resonance imaging (rs-fMRI) data from 17 international sites, and all data are anonymous. The used dataset contains brain networks from 1009 subjects, with 516 (51.14%) being Autism spectrum disorder (ASD) patients (positives). The region definition is based on [Craddock 200 atlas](https://pubmed.ncbi.nlm.nih.gov/21769991/). As the most convenient open-source large-scale dataset, it provides generated brain networks and can be downloaded directly without permission request.


# Download and Preprocess
These scripts will download and preprocess ABIDE dataset.

## Usage

```bash
cd util/abide/

# If you meet time-out error, execute this command repeatly. The script can continue to download from the last failed file.
python 01-fetch_data.py --root_path /path/to/the/save/folder/ --id_file_path subject_IDs.txt --download True

# Generate correlation matrices.
python 02-process_data.py --root_path /path/to/the/save/folder/ --id_file_path subject_IDs.txt

# Generate the final dataset.
python 03-generate_abide_dataset.py --root_path /path/to/the/save/folder/
```

## Obtained Dataset
After using the above code to download and preprocess the data, you should obtain the data file `abide.npy` which is required as input for BrainGB.

The `abide.npy` file contains the following contents:

- **timeseries**: Represents the BOLD time series data for each subject. It's a numpy array with the shape (#sub, #ROI, #timesteps).
  
- > **Label**: Provides the diagnosis label for Autism spectrum disorder for each subject. '0' denotes negative, and '1' indicates positive. It's a numpy array of shape (#sub).
  
- > **corr**: The correlation matrix calculated from BOLD time series data. It's a numpy array with the shape (#sub, #ROIs, #ROIs).
  
- **pcorr**: Represents the partial correlation matrix derived from the BOLD time series data. It's a numpy array with dimensions (#sub, #ROIs, #ROIs).
  
- **site**: Specifies where the data was collected for each subject. It's a numpy array with shape (#sub).

**`Important Note`:** `Label` and `corr matrix` are the actual *inputs* for BrainGB. `Label` represents the target outcome we are interested in predicting, often indicating the diagnosis or condition of a subject in a brain study. `corr matrix` describes the associated Brain Network. If you are considering running BrainGB using your own dataset, it's important to format your Label and corr matrix similarly to ensure compatibility and accurate results. Ensure that `Label` is in a *numpy array* of shape **(#sub)** and `corr matrix` is structured as a *numpy array* with the shape **(#sub, #ROIs, #ROIs)**.

Place the dataset file "abide.npy"  in the `datasets` folder under the `examples` folder (Create the folder if it does not exist).