import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    print(f"\n{description}")
    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"Error: {description} failed")
        sys.exit(1)

def main():
    print("\nATLAS Top Tagging Pipeline - Unsmeared vs Smeared Comparison")
    print("="*60)


    run_command(
        "python train_torch_small_cpu.py",
        "Step 1/7: Training unsmeared model"
    )


    run_command(
        "python train_torch_smeared.py",
        "Step 2/7: Training smeared model"
    )

    run_command(
        "python evaluate_torch.py --checkpoint checkpoints/efn/best_model.pt --type efn --test_data test_split/test_data.npz",
        "Step 3/7: Evaluating unsmeared model"
    )

    run_command(
        "python evaluate_torch.py --checkpoint checkpoints/efn_smeared/best_model.pt --type efn --test_data test_split/test_data_smeared.npz",
        "Step 4/7: Evaluating smeared model"
    )

 
    run_command(
        "python calc_nominal_only.py --checkpoint checkpoints/efn/",
        "Step 5/7: Calculating metrics for unsmeared model"
    )


    run_command(
        "python calc_nominal_only.py --checkpoint checkpoints/efn_smeared/",
        "Step 6/7: Calculating metrics for smeared model"
    )


    run_command(
        "python plot_roc_compare.py --checkpoint1 checkpoints/efn/ --checkpoint2 checkpoints/efn_smeared/ --label1 'Unsmeared' --label2 'Smeared'",
        "Step 7/7: Creating comparison plots"
    )

    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print("Results: plots/roc_compare.png, plots/background_rejection_compare.png")
    print("Checkpoints: checkpoints/efn/, checkpoints/efn_smeared/")

if __name__ == "__main__":
    main()
