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
    print("\nATLAS Top Tagging Pipeline - Evaluation and Plotting")
    print("="*70)

    # Copy models to separate directories to avoid evaluation conflicts
    run_command(
        "mkdir -p checkpoints/efn_teaching_unsmeared checkpoints/efn_teaching_smeared && "
        "cp checkpoints/efn_teaching/best_model_unsmeared.pt checkpoints/efn_teaching_unsmeared/ && "
        "cp checkpoints/efn_teaching/best_model_smeared.pt checkpoints/efn_teaching_smeared/",
        "Step 1/5: Organizing model checkpoints"
    )

    # Evaluate unsmeared teacher model on unsmeared test data
    run_command(
        "python evaluate_torch.py --checkpoint checkpoints/efn_teaching_unsmeared/best_model_unsmeared.pt --type efn --test_data test_split/test_data_unsmeared.npz",
        "Step 2/6: Evaluating teacher (unsmeared) model on unsmeared test data"
    )

    # Evaluate smeared student model on smeared test data
    run_command(
        "python evaluate_torch.py --checkpoint checkpoints/efn_teaching_smeared/best_model_smeared.pt --type efn --test_data test_split/test_data_smeared.npz",
        "Step 3/6: Evaluating student (smeared) model on smeared test data"
    )

    # Calculate metrics for unsmeared model
    run_command(
        "python calc_nominal_only.py --checkpoint checkpoints/efn_teaching_unsmeared/",
        "Step 4/6: Calculating metrics for teacher (unsmeared) model"
    )

    # Calculate metrics for smeared model
    run_command(
        "python calc_nominal_only.py --checkpoint checkpoints/efn_teaching_smeared/",
        "Step 5/6: Calculating metrics for student (smeared) model"
    )

    # Create comparison plots
    run_command(
        "python plot_roc_compare.py --checkpoint1 checkpoints/efn_teaching_unsmeared/ --checkpoint2 checkpoints/efn_teaching_smeared/ --label1 'Teacher (Unsmeared)' --label2 'Student (Smeared)'",
        "Step 6/6: Creating comparison plots"
    )

    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print("Results: plots/roc_compare.png, plots/background_rejection_compare.png")
    print("Checkpoints: checkpoints/efn_teaching/")
    print("  - best_model_unsmeared.pt (Teacher: trained & tested on unsmeared data)")
    print("  - best_model_smeared.pt (Student: trained & tested on smeared data)")
    print("\nNote: Each model evaluated on its corresponding test data type")

if __name__ == "__main__":
    main()
