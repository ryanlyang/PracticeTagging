"""

######################### ATLAS Top Tagging Open Data ##########################

calc.py - This is an example script for calculating performance metrics of
a tagger trained on the ATLAS Top Tagging Open Data set.

IMPORTANT: Performance metrics for all experimental systematic variations
are constrained to be calculated ONLY using jets from the nominal testing
set. This is to ensure that overtraining does not effect the derived uncertainties.

For details of the data set and performance baselines, see:
                       https://cds.cern.ch/record/2825328

Author: Kevin Greif
Last updated 06/24/2024
Written in python 3

################################################################################

"""

from pathlib import Path
import argparse
import copy
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

import utils

########################## Parse Arguments ###########################

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint file, \
                    with all needed predictions stored')
args = parser.parse_args()

################################# SETTINGS #####################################

# Define performance metrics needed for setting systematics. This is a
# dictionary of dictionaries, where each key is a systematic variation and
# each value is a dictionary describing how the calculation should proceed.
# Comment any dictionary out to exclude calculation

# Define some default settings
def_dict = {
    'add_nominal_bkg': False,
    'add_nominal_sig': False,
    'apply_weight': False,
    'weight_bkg': False,
    'extrapolate': False,
}

# Define the performance metrics to calculate
metrics_dict = {
    'nominal': {**def_dict, **{
        'predictions': 'public_test_nominal.npz',
    }},
    'esup': {**def_dict, **{
        'predictions': 'public_esup.npz',
    }},
    'esdown': {**def_dict, **{
        'predictions': 'public_esdown.npz',
    }},
    'cer': {**def_dict, **{
        'predictions': 'public_cer.npz',
    }},
    'cpos': {**def_dict, **{
        'predictions': 'public_cpos.npz',
    }},
    'tfl': {**def_dict, **{
        'predictions': 'public_tfl.npz',
    }},
    'tfj': {**def_dict, **{
        'predictions': 'public_tfj.npz',
    }},
    'teg': {**def_dict, **{
        'predictions': 'public_teg.npz',
    }},
    'tej': {**def_dict, **{
        'predictions': 'public_tej.npz',
    }},
    'bias': {**def_dict, **{
        'predictions': 'public_bias.npz',
    }},
    'ttbar_pythia': {**def_dict, **{
        'predictions': 'public_ttbar_pythia.npz',
        'add_nominal_bkg': True,
        'extrapolate': True,
    }},
    'ttbar_herwig': {**def_dict, **{
        'predictions': 'public_ttbar_herwig.npz',
        'add_nominal_bkg': True,
        'extrapolate': True,
    }},
    'cluster': {**def_dict, **{
        'predictions': 'public_cluster.npz',
        'add_nominal_sig': True,
    }},
    'string': {**def_dict, **{
        'predictions': 'public_string.npz',
        'add_nominal_sig': True,
    }},
    'angular': {**def_dict, **{
        'predictions': 'public_angular.npz',
        'add_nominal_sig': True,
    }},
    'dipole': {**def_dict, **{
        'predictions': 'public_dipole.npz',
        'add_nominal_sig': True,
    }},
    'sig_ISRx2': {**def_dict, **{
        'predictions': 'public_test_nominal.npz',
        'apply_weight': 4,
    }},
    'sig_ISRxp5': {**def_dict, **{
        'predictions': 'public_test_nominal.npz',
        'apply_weight': 9,
    }},
    'sig_FSRx2': {**def_dict, **{
        'predictions': 'public_test_nominal.npz',
        'apply_weight': 6,
    }},
    'sig_FSRxp5': {**def_dict, **{
        'predictions': 'public_test_nominal.npz',
        'apply_weight': 7,
    }},
    'bkg_ISRx2': {**def_dict, **{
        'predictions': 'public_test_nominal.npz',
        'apply_weight': 4,
        'weight_bkg': True,
    }},
    'bkg_ISRxp5': {**def_dict, **{
        'predictions': 'public_test_nominal.npz',
        'apply_weight': 9,
        'weight_bkg': True,
    }},
    'bkg_FSRx2': {**def_dict, **{
        'predictions': 'public_test_nominal.npz',
        'apply_weight': 6,
        'weight_bkg': True,
    }},
    'bkg_FSRxp5': {**def_dict, **{
        'predictions': 'public_test_nominal.npz',
        'apply_weight': 7,
        'weight_bkg': True,
    }},
}

# Define bins of jet pt in which to calculate performance metrics
full_bins = np.linspace(350000, 3150000, 15)

# Define fixed signal efficiency working point at which to evaluate metrics
wp = 0.5

############################## Calculate Metrics ###############################

# Load information on the nominal test set
nom_pfile = np.load(Path(args.checkpoint) / 'public_test_nominal.npz')
nom_preds = nom_pfile['predictions']
nom_labels = nom_pfile['labels']
shower_weights = nom_pfile['shower_weights']
nominal_weights = shower_weights[:,0]  # Nominal MC weight is the first shower weight
nom_pt = nom_pfile['pt']

# Loop through the metrics dict
for name, mdict in metrics_dict.items():

    print(f"\nNow calculating metrics for: {name}")

    # Load .npz file
    pfile = np.load(Path(args.checkpoint) / mdict['predictions'])
    preds = pfile['predictions']
    labels = pfile['labels']
    pt = pfile['pt']

    # Check if we need to append nominal jets to the predictions
    if mdict['add_nominal_bkg'] or mdict['add_nominal_sig']:
        if mdict['add_nominal_sig']:
            target = 1
        elif mdict['add_nominal_bkg']:
            target = 0
        nom_pred_sep = nom_preds[nom_labels == target]
        nom_pt_sep = nom_pt[nom_labels == target]
        preds = np.concatenate((preds, nom_pred_sep))
        labels = np.concatenate((labels, target * np.ones(nom_pred_sep.shape[0])))
        pt = np.concatenate((pt, nom_pt_sep))

    # Check if we need to apply shower weights, can only be done with nominal datasets
    # Otherwise just use a vector of ones as weights
    weights = np.ones(labels.shape)
    if mdict['apply_weight']:
        assert 'nominal' in mdict['predictions']
        var_weights = shower_weights[:,mdict['apply_weight']] / nominal_weights
        if mdict['weight_bkg']:
            weights[nom_labels == 0] = var_weights[nom_labels == 0]
        else:
            weights[nom_labels == 1] = var_weights[nom_labels == 1]

    # Evaluate metrics
    auc = metrics.roc_auc_score(labels, preds, sample_weight=weights)
    discrete_preds = (preds > 0.5).astype(int)
    acc = metrics.accuracy_score(labels, discrete_preds, sample_weight=weights)

    # Evaluate background rejection at fixed signal efficiency working point
    fpr, tpr, thresholds = metrics.roc_curve(labels, preds, sample_weight=weights)
    fprinv = 1 / fpr
    index = np.argmax(tpr > wp)
    br = fprinv[index]

    # Add roc curve to the saved metrics
    save_dict = {
        'tpr': tpr,
        'fprinv': fprinv,
        'thresholds': thresholds,
        'br': br,
        'wp': wp
    }

    # Print metric results
    print('Performance metrics evaluated over testing set:')
    print('AUC score:', auc)
    print('ACC score:', acc)
    print(f'Background rejection at {wp} signal efficiency: {br}')

    # If we are evaluating over SM ttbar datasets, limit binned BR measurement
    # to below 1.5 TeV
    if mdict['extrapolate']:
        high_bin_edge = np.argwhere(full_bins > 1500000)[0][0]
        print("Edge of highest pT bin:", full_bins[high_bin_edge])
        pt_bins = full_bins[:high_bin_edge]
    else:
        pt_bins = full_bins

    # Now calculate background rejection metric in bins of jet pt
    # Initialize array to accept information
    binned_br = np.zeros(len(pt_bins))

    # Loop through bins, since array defines bin edges only go up to
    # length - 1
    for i in range(len(pt_bins)-1):

        # Find indeces of predictions for jets in pt range
        condition = np.logical_and(pt > pt_bins[i], pt < pt_bins[i+1])
        bin_indeces = np.asarray(condition).nonzero()[0]

        # Now take a sub-sample of predictions within the pt bin
        bin_preds = preds[bin_indeces]
        bin_labels = labels[bin_indeces]
        bin_weights = weights[bin_indeces]

        # Now we want to calculate background rejection at working points
        fpr, tpr, thresholds = metrics.roc_curve(bin_labels, bin_preds, sample_weight=bin_weights)
        fprinv = 1 / fpr
        index = np.argmax(tpr > wp)
        binned_br[i] = fprinv[index]

    # Repeat last value for plotting purposes
    binned_br[-1] = fprinv[index]

    # Save the binned background rejection numbers
    save_dict['binned_br'] = binned_br
    save_dict['pt_bins'] = pt_bins
    save_path = Path(args.checkpoint) / (name + "_metrics.npz")
    np.savez(save_path, **save_dict)