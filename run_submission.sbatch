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

# Stop execution on any error
set -e

# Create logs directory for output
mkdir -p logs

echo "=================================================="
echo "Starting Full News Recommendation Pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Node List: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"
echo "=================================================="

# 1. Environment Setup (From Scratch)
echo "[1/5] Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

echo "[2/5] Installing dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
if [ -d "../ebnerd-benchmark" ]; then
    pip install ../ebnerd-benchmark/
fi

# Set thread parallelism environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export POLARS_MAX_THREADS=$SLURM_CPUS_PER_TASK

# 2. Data Preparation & Downloading
echo "[3/5] Downloading and building datasets (MIND + EB-NeRD)..."
make data-large

# 3. Embedding Generation (Accelerated via GPU)
echo "[4/5] Computing BERT embeddings..."
make embed

# 4. Generate Final Submissions for Codabench
echo "[5/5] Generating all 4 Codabench submission files..."
make submission

echo "=================================================="
echo "Pipeline Completed Successfully at $(date)"
echo "Submission ZIP files created in outputs/submissions/:"
ls -lh outputs/submissions/*.zip
echo "=================================================="
