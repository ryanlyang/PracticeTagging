
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint1', type=str, required=True, help='Path to first model checkpoint directory')
parser.add_argument('--checkpoint2', type=str, required=True, help='Path to second model checkpoint directory')
parser.add_argument('--label1', type=str, default='Unsmeared', help='Label for first model')
parser.add_argument('--label2', type=str, default='Smeared', help='Label for second model')
args = parser.parse_args()


checkpoint_path1 = Path(args.checkpoint1)
checkpoint_path2 = Path(args.checkpoint2)

metrics1 = np.load(checkpoint_path1 / 'nominal_metrics.npz')
metrics2 = np.load(checkpoint_path2 / 'nominal_metrics.npz')


plots_dir = Path("./plots")
plots_dir.mkdir(exist_ok=True)


plt.figure(figsize=(8, 6))
plt.plot(metrics1['tpr'], metrics1['fpr'], "-", label=args.label1, color='crimson', linewidth=2)
plt.plot(metrics2['tpr'], metrics2['fpr'], "--", label=args.label2, color='steelblue', linewidth=2)
plt.ylabel(r"False Positive Rate", fontsize=12)
plt.xlabel(r"True Positive Rate (Signal efficiency)", fontsize=12)
plt.legend(fontsize=12, frameon=False)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plots_dir / 'roc_compare.png', dpi=300)
print(f"Saved ROC plot to {plots_dir / 'roc_compare.png'}")


pt_bins1 = metrics1['pt_bins'] / 1000
pt_bins2 = metrics2['pt_bins'] / 1000

plt.figure(figsize=(8, 6))
plt.plot(pt_bins1, metrics1['binned_br'], "-", drawstyle='steps-post',
         label=args.label1, color='crimson', linewidth=2)
plt.plot(pt_bins2, metrics2['binned_br'], "--", drawstyle='steps-post',
         label=args.label2, color='steelblue', linewidth=2)
plt.ylabel(r"Background rejection", fontsize=12)
plt.xlabel(r"Jet $p_T$ [GeV]", fontsize=12)
plt.legend(fontsize=12, frameon=False)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plots_dir / 'background_rejection_compare.png', dpi=300)
print(f"Saved BR plot to {plots_dir / 'background_rejection_compare.png'}")


print(f"\n{args.label1}:")
print(f"  Background rejection at {metrics1['wp']} signal efficiency: {metrics1['br']:.2f}")

print(f"\n{args.label2}:")
print(f"  Background rejection at {metrics2['wp']} signal efficiency: {metrics2['br']:.2f}")
