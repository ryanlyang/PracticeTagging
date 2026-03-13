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
    print("\nATLAS Top Tagging Pipeline - Dual Model Training (Unsmeared vs Smeared)")
    print("="*70)

    # Train both models simultaneously
    run_command(
        "python tester_save_copy.py",
        "Step 1/7: Training both unsmeared and smeared models"
    )

    # Copy models to separate directories to avoid evaluation conflicts
    run_command(
        "mkdir -p checkpoints/efn_teaching_unsmeared checkpoints/efn_teaching_smeared && "
        "cp checkpoints/efn_teaching/best_model_unsmeared.pt checkpoints/efn_teaching_unsmeared/ && "
        "cp checkpoints/efn_teaching/best_model_smeared.pt checkpoints/efn_teaching_smeared/",
        "Step 2/7: Organizing model checkpoints"
    )

    # Evaluate unsmeared model - saves to efn_teaching_unsmeared/
    run_command(
        "python evaluate_torch.py --checkpoint checkpoints/efn_teaching_unsmeared/best_model_unsmeared.pt --type efn --test_data test_split/test_data_smeared.npz",
        "Step 3/7: Evaluating unsmeared model on smeared test data"
    )

    # Evaluate smeared model - saves to efn_teaching_smeared/
    run_command(
        "python evaluate_torch.py --checkpoint checkpoints/efn_teaching_smeared/best_model_smeared.pt --type efn --test_data test_split/test_data_smeared.npz",
        "Step 4/7: Evaluating smeared model on smeared test data"
    )

    # Calculate metrics for unsmeared model
    run_command(
        "python calc_nominal_only.py --checkpoint checkpoints/efn_teaching_unsmeared/",
        "Step 5/7: Calculating metrics for unsmeared model"
    )

    # Calculate metrics for smeared model
    run_command(
        "python calc_nominal_only.py --checkpoint checkpoints/efn_teaching_smeared/",
        "Step 6/7: Calculating metrics for smeared model"
    )

    # Create comparison plots
    run_command(
        "python plot_roc_compare.py --checkpoint1 checkpoints/efn_teaching_unsmeared/ --checkpoint2 checkpoints/efn_teaching_smeared/ --label1 'Unsmeared Model' --label2 'Smeared Model'",
        "Step 7/7: Creating comparison plots"
    )

    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print("Results: plots/roc_compare.png, plots/background_rejection_compare.png")
    print("Checkpoints: checkpoints/efn_teaching/")
    print("  - best_model_unsmeared.pt (trained on unsmeared data)")
    print("  - best_model_smeared.pt (trained on smeared data)")
    print("\nBoth models evaluated on smeared test data")

if __name__ == "__main__":
    main()
