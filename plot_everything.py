"""

######################### ATLAS Top Tagging Open Data ##########################

plot_everything.py - This script makes plots of the performance metrics of a tagger
compared to benchmark taggers, and the systematic uncertainties in these
performance metrics, again compared to benchmark taggers.

For details of the data set and performance baselines, see:
                       https://cds.cern.ch/record/2825328

Author: Kevin Greif
Last updated 06/20/2024
Written in python 3

################################################################################

"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

########################## Parse Arguments ###########################

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint file, \
                    with all needed metrics stored')
parser.add_argument('--netName', type=str, help='The label for the network of interest')
args = parser.parse_args()

############################## Load Metrics ####################################

store_path = Path(args.checkpoint)
nominal = np.load(store_path / 'nominal_metrics.npz')
esup = np.load(store_path / 'esup_metrics.npz')
esdown = np.load(store_path / 'esdown_metrics.npz')
cer = np.load(store_path / 'cer_metrics.npz')
cpos = np.load(store_path / 'cpos_metrics.npz')
teg = np.load(store_path / 'teg_metrics.npz')
tej = np.load(store_path / 'tej_metrics.npz')
tfl = np.load(store_path / 'tfl_metrics.npz')
tfj = np.load(store_path / 'tfj_metrics.npz')
bias = np.load(store_path / 'bias_metrics.npz')
ttbar_pythia = np.load(store_path / 'ttbar_pythia_metrics.npz')
ttbar_herwig = np.load(store_path / 'ttbar_herwig_metrics.npz')
cluster = np.load(store_path / 'cluster_metrics.npz')
string = np.load(store_path / 'string_metrics.npz')
angular = np.load(store_path / 'angular_metrics.npz')
dipole = np.load(store_path / 'dipole_metrics.npz')
sig_ISRx2 = np.load(store_path / 'sig_ISRx2_metrics.npz')
sig_ISRxp5 = np.load(store_path / 'sig_ISRxp5_metrics.npz')
sig_FSRx2 = np.load(store_path / 'sig_FSRx2_metrics.npz')
sig_FSRxp5 = np.load(store_path / 'sig_FSRxp5_metrics.npz')
bkg_ISRx2 = np.load(store_path / 'bkg_ISRx2_metrics.npz')
bkg_ISRxp5 = np.load(store_path / 'bkg_ISRxp5_metrics.npz')
bkg_FSRx2 = np.load(store_path / 'bkg_FSRx2_metrics.npz')
bkg_FSRxp5 = np.load(store_path / 'bkg_FSRxp5_metrics.npz')

# Get pT binning
pt_bins = nominal['pt_bins']
plot_bins = pt_bins / 1000
bin_centers = (plot_bins[1:] + plot_bins[:-1]) / 2

###################### ROC and background rejection plots ######################

# Load hlDNN and ParticleNet ROC curves and BRs
bench_path = Path("./benchmarks")
hlDNN_roc = np.load(bench_path / 'hldnn_roc.npz')
hlDNN_brs = np.load(bench_path / 'hldnn_brs.npz')
ParticleNet_roc = np.load(bench_path / 'pnet_roc.npz')
ParticleNet_brs = np.load(bench_path / 'pnet_brs.npz')

# Plot ROC curve
roc = plt.figure()
ax = roc.add_subplot(111)
ax.plot(hlDNN_roc['tpr'], 1 / hlDNN_roc['fpr'], "-", label='hlDNN', color='#b8b8b8')
ax.plot(ParticleNet_roc['tpr'], 1 / ParticleNet_roc['fpr'], "--", label='ParticleNet', color='steelblue')
ax.plot(nominal['tpr'], nominal['fprinv'], "-", color='crimson', label=args.netName)
ax.set_yscale('log')
ax.set_ylabel(r"Background rejection", fontsize=12)
ax.set_xlabel(r"Signal efficiency", fontsize=12)
ax.legend(fontsize=12, frameon=False)
roc.tight_layout()
roc.savefig('./plots/roc.png', dpi=300)

# Plot background rejection
# Only do this if the pT binning matches the benchmarks and the nominal wp is 0.5
if np.all(pt_bins == np.linspace(350000, 3150000, 15)) and nominal['wp'] == 0.5:

    # Make plot
    br = plt.figure()
    ax = br.add_subplot(111)
    ax.plot(plot_bins, hlDNN_brs['total_binned_br_50'], "-", drawstyle='steps-post', label='hlDNN', color='#b8b8b8')
    ax.plot(plot_bins, ParticleNet_brs['total_binned_br_50'], "--", drawstyle='steps-post', label='ParticleNet', color='steelblue')
    ax.plot(plot_bins, nominal['binned_br'], "-", drawstyle='steps-post', label=args.netName, color='crimson')
    ax.set_ylabel(r"Background rejection", fontsize=12)
    ax.set_xlabel(r"Jet $p_T$ [GeV]", fontsize=12)
    ax.legend(fontsize=12, frameon=False)
    br.tight_layout()
    br.savefig('./plots/background_rejection.png', dpi=300)

###################### Systematics plots ###############################

# Calculate cluster systematic uncertainties
print(nominal['binned_br'])
print(cpos['binned_br'])
esup_uncert = np.abs(esup['binned_br'] / nominal['binned_br'] - 1)
esdown_uncert = np.abs(esdown['binned_br'] / nominal['binned_br'] - 1)
cer_uncert = np.abs(cer['binned_br'] / nominal['binned_br'] - 1)
cpos_uncert = np.abs(cpos['binned_br'] / nominal['binned_br'] - 1)

# Energy scale is the envelope of esup and esdown
es_uncert = np.maximum(esup_uncert, esdown_uncert)

# Cluster uncertainty is the quadrature sum of scale, resolution, position, and efficiency
cluster_uncert = np.sqrt(es_uncert**2 + cer_uncert**2 + cpos_uncert**2)

# Make plot
clus = plt.figure()
ax = clus.add_subplot(111)
ax.plot(bin_centers, 100*es_uncert[:-1], "*", label='Scale')
ax.plot(bin_centers, 100*cer_uncert[:-1], ">", label='Resolution')
ax.plot(bin_centers, 100*cpos_uncert[:-1], "<", label='Position')
ax.plot(plot_bins, 100*cluster_uncert, "-", drawstyle='steps-post', color='#000000', label='Total')
ax.fill_between(plot_bins, 100*cluster_uncert, step='post', color='#c8ccfa', alpha=0.2)
ax.legend(loc='upper right', fontsize=12, frameon=False)
ax.set_ylabel('$\epsilon_{bkg}^{-1}$ Relative Uncertainty [%]', fontsize=12)
ax.set_xlabel(r'Large-R Jet $p_T$ [GeV]', fontsize=12)
clus.tight_layout()
clus.savefig('./plots/cluster_uncertainties.png', dpi=300)

# Calculate track systematic uncertainties
teg_uncert = np.abs(teg['binned_br'] / nominal['binned_br'] - 1)
tej_uncert = np.abs(tej['binned_br'] / nominal['binned_br'] - 1)
tfl_uncert = np.abs(tfl['binned_br'] / nominal['binned_br'] - 1)
tfj_uncert = np.abs(tfj['binned_br'] / nominal['binned_br'] - 1)
bias_uncert = np.abs(bias['binned_br'] / nominal['binned_br'] - 1)

# Efficiency and fake rate are the envelopes over inclusive and within jet variations
eff_uncert = np.maximum(teg_uncert, tej_uncert)
fake_uncert = np.maximum(tfl_uncert, tfj_uncert)

# Track uncertainty is the quadrature sum of efficiency, fake rate, and bias
track_uncert = np.sqrt(eff_uncert**2 + fake_uncert**2 + bias_uncert**2)

# Make plot
track = plt.figure()
ax = track.add_subplot(111)
ax.plot(bin_centers, 100*eff_uncert[:-1], "*", label='Efficiency')
ax.plot(bin_centers, 100*fake_uncert[:-1], ">", label='Fake rate')
ax.plot(bin_centers, 100*bias_uncert[:-1], "<", label='Bias')
ax.plot(plot_bins, 100*track_uncert, "-", drawstyle='steps-post', color='#000000', label='Total')
ax.fill_between(plot_bins, 100*track_uncert, step='post', color='#c8ccfa', alpha=0.2)
ax.legend(loc='upper right', fontsize=12, frameon=False)
ax.set_ylabel('$\epsilon_{bkg}^{-1}$ Relative Uncertainty [%]', fontsize=12)
ax.set_xlabel(r'Large-R Jet $p_T$ [GeV]', fontsize=12)
track.tight_layout()
track.savefig('./plots/track_uncertainties.png', dpi=300)

# Calculate modeling uncertainties
sig_model = np.abs(ttbar_herwig['binned_br'] / ttbar_pythia['binned_br'] - 1)
bkg_ps = np.abs(dipole['binned_br'] / angular['binned_br'] - 1)
bkg_had = np.abs(cluster['binned_br'] / string['binned_br'] - 1)

# Background modeling uncertainty is quadrature sum of parton shower and hadronization
bkg_model = np.sqrt(bkg_ps**2 + bkg_had**2)
total_model = np.sqrt(sig_model**2 + bkg_model**2)

# Make plot
model = plt.figure()
ax = model.add_subplot(111)
ax.plot(bin_centers, 100*sig_model[:-1], "*", label='Signal modeling')
ax.plot(bin_centers, 100*bkg_ps[:-1], ">", label='Bkg parton shower')
ax.plot(bin_centers, 100*bkg_had[:-1], "<", label='Bkg hadronization')
ax.plot(plot_bins, 100*total_model, "-", drawstyle='steps-post', color='#000000', label='Total')
ax.fill_between(plot_bins, 100*total_model, step='post', color='#c8ccfa', alpha=0.2)
ax.legend(loc='upper right', fontsize=12, frameon=False)
ax.set_ylabel('$\epsilon_{bkg}^{-1}$ Relative Uncertainty [%]', fontsize=12)
ax.set_xlabel(r'Large-R Jet $p_T$ [GeV]', fontsize=12)
model.tight_layout()
model.savefig('./plots/modeling_uncertainties.png', dpi=300)

# Calculate scale uncertainties
sig_ISRx2_uncert = np.abs(sig_ISRx2['binned_br'] / nominal['binned_br'] - 1)
sig_ISRxp5_uncert = np.abs(sig_ISRxp5['binned_br'] / nominal['binned_br'] - 1)
sig_FSRx2_uncert = np.abs(sig_FSRx2['binned_br'] / nominal['binned_br'] - 1)
sig_FSRxp5_uncert = np.abs(sig_FSRxp5['binned_br'] / nominal['binned_br'] - 1)
bkg_ISRx2_uncert = np.abs(bkg_ISRx2['binned_br'] / nominal['binned_br'] - 1)
bkg_ISRxp5_uncert = np.abs(bkg_ISRxp5['binned_br'] / nominal['binned_br'] - 1)
bkg_FSRx2_uncert = np.abs(bkg_FSRx2['binned_br'] / nominal['binned_br'] - 1)
bkg_FSRxp5_uncert = np.abs(bkg_FSRxp5['binned_br'] / nominal['binned_br'] - 1)

# For each scale, take envelope of up and down variations
sig_ISR_uncert = np.maximum(sig_ISRx2_uncert, sig_ISRxp5_uncert)
sig_FSR_uncert = np.maximum(sig_FSRx2_uncert, sig_FSRxp5_uncert)
bkg_ISR_uncert = np.maximum(bkg_ISRx2_uncert, bkg_ISRxp5_uncert)
bkg_FSR_uncert = np.maximum(bkg_FSRx2_uncert, bkg_FSRxp5_uncert)

# Total scale uncertainty is the quadrature sum
scale_uncert = np.sqrt(sig_ISR_uncert**2 + sig_FSR_uncert**2 + bkg_ISR_uncert**2 + bkg_FSR_uncert**2)

# Make plot
scale = plt.figure()
ax = scale.add_subplot(111)
ax.plot(bin_centers, 100*sig_ISR_uncert[:-1], "*", label='Signal ISR')
ax.plot(bin_centers, 100*sig_FSR_uncert[:-1], ">", label='Signal FSR')
ax.plot(bin_centers, 100*bkg_ISR_uncert[:-1], "<", label='Bkg ISR')
ax.plot(bin_centers, 100*bkg_FSR_uncert[:-1], "^", label='Bkg FSR')
ax.plot(plot_bins, 100*scale_uncert, "-", drawstyle='steps-post', color='#000000', label='Total')
ax.fill_between(plot_bins, 100*scale_uncert, step='post', color='#c8ccfa', alpha=0.2)
ax.legend(loc='upper right', fontsize=12, frameon=False)
ax.set_ylabel('$\epsilon_{bkg}^{-1}$ Relative Uncertainty [%]', fontsize=12)
ax.set_xlabel(r'Large-R Jet $p_T$ [GeV]', fontsize=12)
scale.tight_layout()
scale.savefig('./plots/scale_uncertainties.png', dpi=300)

# Total uncertainty is the quadrature sum of all uncertainties
total_uncert = np.sqrt(cluster_uncert**2 + track_uncert**2 + total_model**2 + scale_uncert**2)

# Make plot of total uncertainty
total = plt.figure()
ax = total.add_subplot(111)
ax.plot(plot_bins, 100*cluster_uncert, "-", drawstyle='steps-post', label='Cluster')
ax.plot(plot_bins, 100*track_uncert, "-", drawstyle='steps-post', label='Track')
ax.plot(plot_bins, 100*sig_model, "-", drawstyle='steps-post', label='Signal modeling')
ax.plot(plot_bins, 100*bkg_model, "-", drawstyle='steps-post', label='Background modeling')
ax.plot(plot_bins, 100*scale_uncert, "-", drawstyle='steps-post', label='Scale')
ax.plot(plot_bins, 100*total_uncert, "-", drawstyle='steps-post', color='#000000', label='Total')
ax.fill_between(plot_bins, 100*total_uncert, step='post', color='#c8ccfa', alpha=0.2)
ax.legend(loc='upper right', fontsize=12, frameon=False)
ax.set_ylabel('$\epsilon_{bkg}^{-1}$ Relative Uncertainty [%]', fontsize=12)
ax.set_xlabel(r'Large-R Jet $p_T$ [GeV]', fontsize=12)
total.tight_layout()
total.savefig('./plots/total_uncertainties.png', dpi=300)

