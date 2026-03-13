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

    # Train all three models
    run_command(
        "python tester_teach_copy.py",
        "Step 1/9: Training teacher, taught student, and untaught student"
    )

    # Copy models to separate directories to avoid evaluation conflicts
    run_command(
        "mkdir -p checkpoints/efn_teaching_unsmeared checkpoints/efn_teaching_smeared checkpoints/efn_teaching_untaught && "
        "cp checkpoints/efn_teaching/best_model_unsmeared.pt checkpoints/efn_teaching_unsmeared/ && "
        "cp checkpoints/efn_teaching/best_model_smeared.pt checkpoints/efn_teaching_smeared/ && "
        "cp checkpoints/efn_teaching/best_model_untaught.pt checkpoints/efn_teaching_untaught/",
        "Step 2/9: Organizing model checkpoints"
    )

    # Evaluate unsmeared teacher model on unsmeared test data
    run_command(
        "python evaluate_torch.py --checkpoint checkpoints/efn_teaching_unsmeared/best_model_unsmeared.pt --type efn --test_data test_split/test_data_unsmeared.npz",
        "Step 3/9: Evaluating teacher (unsmeared) on unsmeared test data"
    )

    # Evaluate taught student model on smeared test data
    run_command(
        "python evaluate_torch.py --checkpoint checkpoints/efn_teaching_smeared/best_model_smeared.pt --type efn --test_data test_split/test_data_smeared.npz",
        "Step 4/9: Evaluating taught student (smeared + KD) on smeared test data"
    )

    # Evaluate untaught student model on smeared test data
    run_command(
        "python evaluate_torch.py --checkpoint checkpoints/efn_teaching_untaught/best_model_untaught.pt --type efn --test_data test_split/test_data_smeared.npz",
        "Step 5/9: Evaluating untaught student (smeared only) on smeared test data"
    )

    # Calculate metrics for teacher model
    run_command(
        "python calc_nominal_only.py --checkpoint checkpoints/efn_teaching_unsmeared/",
        "Step 6/9: Calculating metrics for teacher model"
    )

    # Calculate metrics for taught student model
    run_command(
        "python calc_nominal_only.py --checkpoint checkpoints/efn_teaching_smeared/",
        "Step 7/9: Calculating metrics for taught student model"
    )

    # Calculate metrics for untaught student model
    run_command(
        "python calc_nominal_only.py --checkpoint checkpoints/efn_teaching_untaught/",
        "Step 8/9: Calculating metrics for untaught student model"
    )

    # Create comparison plots with three curves
    run_command(
        "python plot_roc_compare_three.py --checkpoint1 checkpoints/efn_teaching_unsmeared/ --checkpoint2 checkpoints/efn_teaching_smeared/ --checkpoint3 checkpoints/efn_teaching_untaught/ --label1 'Teacher (Unsmeared)' --label2 'Taught Student (Smeared + KD)' --label3 'Untaught Student (Smeared)'",
        "Step 9/9: Creating three-way comparison plots"
    )

    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print("Results: plots/roc_compare_three.png, plots/background_rejection_compare_three.png")
    print("Checkpoints: checkpoints/efn_teaching/")
    print("  - best_model_unsmeared.pt (Teacher: trained on unsmeared, tested on unsmeared)")
    print("  - best_model_smeared.pt (Taught Student: trained on smeared + KD, tested on smeared)")
    print("  - best_model_untaught.pt (Untaught Student: trained on smeared only, tested on smeared)")
    print("\nNote: Three-way comparison shows the effect of knowledge distillation")

if __name__ == "__main__":
    main()
