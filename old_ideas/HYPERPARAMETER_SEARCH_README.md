# Knowledge Distillation Hyperparameter Search - SLURM Workflow

## Overview

This workflow optimizes Knowledge Distillation (KD) hyperparameters for transformer-based top tagging models:

- **Temperature Annealing**: Start with higher temperature (softer teaching), decrease over training
- **Alpha Scheduling**: Start with more KD weight, gradually shift toward hard labels

**Key Optimization**: Teacher and baseline models are trained ONCE and reused across all hyperparameter runs, saving ~66% training time per job!

## Workflow

### Step 1: Setup Environment (one-time)

```bash
./setup_environment.sh
```

This creates the `atlas_kd` conda environment with all dependencies. See [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) for details.

### Step 2: Train Shared Models (one-time)

```bash
sbatch train_shared_models.sh
```

**What this does:**
- Trains the teacher model (offline/unsmeared data, no hyperparameters)
- Trains the baseline model (HLT/smeared data, no KD, no hyperparameters)
- Saves models to `checkpoints/transformer_search/shared_models/`
- These models will be reused for ALL hyperparameter search runs

**Wait for completion:**
```bash
# Check job status
squeue -u $USER

# Watch progress
tail -f transformer_logs/train_shared_*.out
```

### Step 3: Run Hyperparameter Search

```bash
./submit_all_hyperparameter_jobs.sh
```

**What this does:**
- Verifies shared models exist
- Submits ~144 SLURM jobs (one per hyperparameter configuration)
- Each job:
  - Loads pre-trained teacher and baseline models
  - Trains ONLY the KD student model with specific temperature/alpha schedule
  - Saves results to `checkpoints/transformer_search/<run_name>/`
  - Logs metrics to `checkpoints/transformer_search/hyperparameter_search_results.txt`

**Monitor progress:**
```bash
# Check all jobs
squeue -u $USER

# Watch a specific job
tail -f transformer_logs/transformer_<job_id>.out

# Count completed jobs
ls checkpoints/transformer_search/ | grep -E "baseline_|tempanneal_|alphasched_|combined_" | wc -l
```

### Step 4: Analyze Results

```bash
# Once jobs complete, analyze and rank results
python analyze_hyperparameter_results.py
```

This shows:
- Top 10 configurations by Background Rejection @ 50% signal efficiency
- Statistics across all runs
- Best configuration details

## File Structure

```
PracticeTagging/
├── data/
│   └── test.h5                              # Training data
├── transformer_runner.py                    # Main training script
├── train_shared_models.sh                   # SLURM: Train teacher + baseline once
├── run_transformer_single.sh                # SLURM: Single hyperparameter run
├── submit_all_hyperparameter_jobs.sh        # Submit all search jobs
├── analyze_hyperparameter_results.py        # Analyze and rank results
├── setup_environment.sh                     # Environment setup script
├── requirements.txt                         # Python dependencies
├── checkpoints/
│   └── transformer_search/
│       ├── shared_models/
│       │   ├── teacher.pt                   # Pre-trained teacher (reused)
│       │   └── baseline.pt                  # Pre-trained baseline (reused)
│       ├── baseline_T7.0_A0.5/              # Hyperparameter run results
│       ├── tempanneal_T7.0to2.0_A0.5/
│       ├── ...
│       └── hyperparameter_search_results.txt  # Summary of all runs
└── transformer_logs/
    ├── train_shared_12345.out               # Shared models training log
    ├── transformer_12346.out                # Individual search job logs
    └── ...
```

## Hyperparameter Grid

### Temperature
- **Values**: 3.0, 5.0, 7.0, 10.0
- **Annealing targets**: 1.0, 2.0
- **Effect**: Higher T = softer teaching, lower T = sharper distributions

### Alpha (KD weight)
- **Values**: 0.3, 0.5, 0.7, 0.9
- **Scheduling targets**: 0.1, 0.3
- **Effect**: Higher α = more KD, lower α = more hard labels

### Configurations
1. **Baseline** (16 runs): Constant T × constant α
2. **Temperature Annealing** (~32 runs): T decreases, α constant
3. **Alpha Scheduling** (~32 runs): T constant, α decreases
4. **Combined** (~64 runs): Both T and α vary

**Total**: ~144 configurations

## Time and Space Savings

### Time Savings

Without shared models:
- Each job trains: Teacher (~30 min) + Baseline (~30 min) + Student (~30 min) = 90 min
- 144 jobs × 90 min = 216 hours of compute time

With shared models:
- Shared training: 1 job × 90 min = 90 min (one-time)
- Each search job: Student only (~30 min)
- 144 jobs × 30 min = 72 hours of compute time
- **Total**: 90 min + 72 hours = 73.5 hours
- **Time savings**: 142.5 hours (66% reduction!)

### Space Savings

Without model skipping:
- Each job saves: Teacher (~500 MB) + Baseline (~500 MB) + Student (~500 MB) = 1.5 GB
- 144 jobs × 1.5 GB = 216 GB of storage

With `--skip_save_models`:
- Shared models: Teacher (~500 MB) + Baseline (~500 MB) = 1 GB (one-time)
- Each search job: Only predictions and metrics (~1 MB)
- 144 jobs × 1 MB = 144 MB
- **Total**: 1 GB + 144 MB = 1.14 GB
- **Space savings**: 214.86 GB (99% reduction!)

Note: Only the shared teacher and baseline models are saved. All hyperparameter search runs save predictions, ROC curves, and metrics but not model weights.

## Metrics

### Primary: Background Rejection @ 50% Signal Efficiency
- How many background events are rejected while keeping 50% of signal
- Higher is better
- Physics-relevant metric for jet tagging

### Secondary: AUC (Area Under ROC Curve)
- Single-number classifier quality measure
- Range: 0.5 (random) to 1.0 (perfect)
- Higher is better

## Manual Single Run

To test a specific configuration without the full search:

```bash
# Example: Temperature annealing 7.0→2.0, alpha constant 0.5
sbatch --export=ALL,TEMP_INIT=7.0,TEMP_FINAL=2.0,ALPHA_INIT=0.5,RUN_NAME="my_test" \
       run_transformer_single.sh
```

Or run locally (no SLURM):
```bash
conda activate atlas_kd
python transformer_runner.py \
    --run_name "my_test" \
    --temp_init 7.0 \
    --temp_final 2.0 \
    --alpha_init 0.5 \
    --teacher_checkpoint checkpoints/transformer_search/shared_models/teacher.pt \
    --baseline_checkpoint checkpoints/transformer_search/shared_models/baseline.pt \
    --device cpu
```

## Troubleshooting

### "Shared models not found" error

**Problem**: Tried to run hyperparameter search before training shared models.

**Solution**:
```bash
sbatch train_shared_models.sh
# Wait for completion
squeue -u $USER
# Then run search
./submit_all_hyperparameter_jobs.sh
```

### Jobs fail immediately

**Check logs**:
```bash
cat transformer_logs/transformer_*.err
```

Common issues:
- Environment not activated (check conda in script)
- Missing data files (check path in transformer_runner.py)
- Shared models missing (train them first)

### Out of memory errors

Reduce batch size in transformer_runner.py CONFIG section, or request more memory in the SLURM script.

## Customizing the Search

Edit `submit_all_hyperparameter_jobs.sh`:

```bash
# Modify temperature grid
TEMPERATURES=(5.0 7.0 10.0 15.0)  # Add 15.0

# Modify alpha grid
ALPHAS=(0.2 0.4 0.6 0.8)  # Different values

# Modify annealing targets
TEMP_ANNEAL_FINALS=(0.5 1.0 2.0)  # Add 0.5
```

## Next Steps

After finding the best hyperparameters:
1. Review top configurations from `analyze_hyperparameter_results.py`
2. Check individual plots in `checkpoints/transformer_search/<best_run>/results.png`
3. Use best configuration for final model training
4. Compare with EFN models from other scripts
