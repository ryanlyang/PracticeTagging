

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, help='Path to checkpoint directory')
parser.add_argument('--netName', type=str, default='My Model', help='Name of your model')
args = parser.parse_args()

checkpoint_path = Path(args.checkpoint)
nominal = np.load(checkpoint_path / 'nominal_metrics.npz')


bench_path = Path("./benchmarks")
hlDNN_roc = np.load(bench_path / 'hldnn_roc.npz')
ParticleNet_roc = np.load(bench_path / 'pnet_roc.npz')


plots_dir = Path("./plots")
plots_dir.mkdir(exist_ok=True)


plt.figure(figsize=(8, 6))
plt.plot(hlDNN_roc['tpr'], hlDNN_roc['fpr'], "-", label='hlDNN', color='#b8b8b8')
plt.plot(ParticleNet_roc['tpr'], ParticleNet_roc['fpr'], "--", label='ParticleNet', color='steelblue')
plt.plot(nominal['tpr'], nominal['fpr'], "-", color='crimson', label=args.netName)
plt.ylabel(r"False Positive Rate", fontsize=12)
plt.xlabel(r"True Positive Rate (Signal efficiency)", fontsize=12)
plt.legend(fontsize=12, frameon=False)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plots_dir / 'roc_simple.png', dpi=300)
print(f"Saved ROC plot to {plots_dir / 'roc_simple.png'}")


pt_bins = nominal['pt_bins'] / 1000  
bin_centers = (pt_bins[1:] + pt_bins[:-1]) / 2

plt.figure(figsize=(8, 6))
plt.plot(pt_bins, nominal['binned_br'], "-", drawstyle='steps-post', label=args.netName, color='crimson', linewidth=2)
plt.ylabel(r"Background rejection", fontsize=12)
plt.xlabel(r"Jet $p_T$ [GeV]", fontsize=12)
plt.legend(fontsize=12, frameon=False)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plots_dir / 'background_rejection_simple.png', dpi=300)
print(f"Saved BR plot to {plots_dir / 'background_rejection_simple.png'}")


print(f"\nSummary for {args.netName}:")
print(f"  Background rejection at {nominal['wp']} signal efficiency: {nominal['br']:.2f}")
