"""

######################### ATLAS Top Tagging Open Data ##########################

calc_nominal_only.py - Simplified script for calculating performance metrics
when you only have nominal test data (no systematic variations).

Modified from calc.py to work with limited data.

################################################################################

"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

########################## Parse Arguments ###########################

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint directory')
args = parser.parse_args()

################################# SETTINGS #####################################

full_bins = np.linspace(350000, 3150000, 15)

wp = 0.5

############################## Calculate Metrics ###############################

print("\nCalculating metrics for: nominal")

checkpoint_path = Path(args.checkpoint)
nom_pfile = np.load(checkpoint_path / 'public_test_nominal.npz')
preds = nom_pfile['predictions']
labels = nom_pfile['labels']
pt = nom_pfile['pt']


weights = np.ones(labels.shape)


auc = metrics.roc_auc_score(labels, preds, sample_weight=weights)
discrete_preds = (preds > 0.5).astype(int)
acc = metrics.accuracy_score(labels, discrete_preds, sample_weight=weights)

fpr, tpr, thresholds = metrics.roc_curve(labels, preds, sample_weight=weights)
fprinv = 1 / fpr
index = np.argmax(tpr > wp)
br = fprinv[index]


save_dict = {
    'tpr': tpr,
    'fpr': fpr,
    'fprinv': fprinv,
    'thresholds': thresholds,
    'br': br,
    'wp': wp
}


print('Performance metrics evaluated over testing set:')
print('AUC score:', auc)
print('ACC score:', acc)
print(f'Background rejection at {wp} signal efficiency: {br}')


pt_bins = full_bins


binned_br = np.zeros(len(pt_bins))


for i in range(len(pt_bins)-1):
    condition = np.logical_and(pt > pt_bins[i], pt < pt_bins[i+1])
    bin_indices = np.asarray(condition).nonzero()[0]

    bin_preds = preds[bin_indices]
    bin_labels = labels[bin_indices]
    bin_weights = weights[bin_indices]

  
    if len(bin_preds) > 0:  
        fpr, tpr, thresholds = metrics.roc_curve(bin_labels, bin_preds, sample_weight=bin_weights)
        fprinv = 1 / fpr
        index = np.argmax(tpr > wp)
        binned_br[i] = fprinv[index]
    else:
        binned_br[i] = 0 


binned_br[-1] = binned_br[-2] if binned_br[-2] > 0 else fprinv[index]


save_dict['binned_br'] = binned_br
save_dict['pt_bins'] = pt_bins
save_path = checkpoint_path / "nominal_metrics.npz"
np.savez(save_path, **save_dict)

print(f"\nMetrics saved to {save_path}")

