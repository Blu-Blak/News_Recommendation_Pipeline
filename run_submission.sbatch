#!/bin/bash
#SBATCH -A research
#SBATCH --job-name=news_rec_pipeline
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-user=dhruv.malik@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --time=05:00:00

# Exit immediately if a command exits with a non-zero status
set -e

# Create logs directory for SLURM output
mkdir -p logs

echo "=================================================="
echo "Starting News Recommendation Pipeline Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node List: $SLURM_NODELIST"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Start Time: $(date)"
echo "=================================================="

# Activate virtual environment if available
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "../venv" ]; then
    source ../venv/bin/activate
fi

# Ensure high thread usage for numpy/polars/faiss
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export POLARS_MAX_THREADS=$SLURM_CPUS_PER_TASK

# 1. Download & build large datasets if missing
echo "Building datasets..."
make data-large

# 2. Compute BERT embeddings for MIND test articles
echo "Generating embeddings..."
make embed

# 3. Generate Codabench submissions (MIND & EB-NeRD for Lexical and Semantic)
echo "Generating final prediction files..."
make submission

echo "=================================================="
echo "Job Finished Successfully at $(date)"
echo "Outputs stored in outputs/submissions/"
echo "=================================================="
