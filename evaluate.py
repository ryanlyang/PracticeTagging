"""

######################### ATLAS Top Tagging Open Data ##########################

evaluate.py - This is an example script for evaluating constituent based taggers
on the ATLAS Top Tagging Open Data set. Model predictions are saved within
the checkpoint directory.

For details of the data set and performance baselines, see:
                       https://cds.cern.ch/record/2825328

Author: Kevin Greif
Last updated 06/24/2024
Written in python 3

################################################################################

"""

from pathlib import Path
import argparse

# Plotting imports
import numpy as np

# ML imports
import tensorflow as tf
from tensorflow.data import Dataset
from energyflow.archs import EFN, PFN
import sklearn.metrics as metrics

# Custom imports
import utils

################################## PARSE ####################################

# Create an argument parser
parser = argparse.ArgumentParser(description='Test a tagger on the ATLAS Top Tagging Open Data set')
parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint file')
parser.add_argument('--data', type=str, help='Path to the directory containing data files')
parser.add_argument('--type', type=str, help='Type of tagger to test', choices=['dnn', 'efn', 'pfn', 'hldnn'])
parser.add_argument('--store_weights', action='store_true', help='Store shower weights in the output file')
parser.add_argument('--store_numbers', action='store_true', help='Store shower numbers in the output file')

# Parse the command line arguments
args = parser.parse_args()

################################# SETTINGS #####################################

# Paths to data files. Point this to local directory containing the data files
# in sub-directories
data_path = Path(args.data)

# Make glob of the testing set files
test_files = sorted(list(data_path.glob("*.h5")))

# Max constituents to consider in tagger testing (must be <= 200)
max_constits = 80

# Testing parameters
batch_size = 256

# The maximum number of jets to use in testing
max_jets = 3000000

########################### Data Preparation ###################################
print("Read data and prepare for tagger testing")

# Load data using the functions in preprocessing.py
test_data, test_labels, _, shower_weights, test_jet_pt, test_numbers = utils.load_from_files(
    test_files,
    max_jets=max_jets,
    get_hl=True if args.type == 'hldnn' else False,
    use_train_weights=False,
    use_shower_weights=args.store_weights,
    use_numbers=args.store_numbers,
    max_constits=max_constits
)

# Find the number of data features
num_data_features = test_data.shape[-1]

# Make dataset, no weights necessary since we are not training
if args.type == 'efn':

    # For EFN, we need to split the data into angular / energy components, then
    # package these into the dataset as done in train.py
    test_angular = test_data[:,:,0:2]
    test_pt = test_data[:,:,2]
    test_lists = [test_pt, test_angular, test_labels]
    test_sets = tuple([Dataset.from_tensor_slices(i).batch(batch_size) for i in test_lists])
    test_data = Dataset.zip(test_sets[:2])
    test_dataset = Dataset.zip((test_data,) + test_sets[2:])

# For DNN, need to flatten the constituents data
elif args.type == 'dnn':
    test_data = test_data.reshape(-1, max_constits * num_data_features)
    test_dataset = Dataset.from_tensor_slices((test_data, test_labels)).batch(batch_size)

# Else just make the dataset
else:
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(batch_size)

######################### Load and evaluate tagger #############################

# Load the tagger from the checkpoint
model = tf.keras.models.load_model(args.checkpoint)

# Evaluate
predictions = model.predict(test_dataset, batch_size=batch_size)[:,0]
discrete_predictions = (predictions > 0.5).astype(int)

############################# Save predictions ###############################

# Drop any NaN predictions
nan_mask = np.isnan(predictions)
predictions = predictions[~nan_mask]
test_labels = test_labels[~nan_mask]
test_jet_pt = test_jet_pt[~nan_mask]
if args.store_weights:
    shower_weights = shower_weights[~nan_mask]
if args.store_numbers:
    test_numbers = test_numbers[~nan_mask]

# Save the data
save_path = Path(args.checkpoint) / (str(data_path.name) + ".npz")
save_dict = {'labels': test_labels, 'predictions': predictions, 'pt': test_jet_pt}
if args.store_weights:
    save_dict['shower_weights'] = shower_weights
if args.store_numbers:
    save_dict['numbers'] = test_numbers
np.savez(save_path, **save_dict)