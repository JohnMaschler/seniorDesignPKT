#!/bin/bash
#
#SBATCH --job-name=train_model
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8         # Allocate 8 CPU cores
#SBATCH --mem=24G                 # Allocate 24GB system memory
#SBATCH --nodes=1                 # Run on a single node
#SBATCH --gres=gpu:1              # Request 1 GPU (any available)
#SBATCH --output=train_model-%j.out  # Save output logs with job ID
#SBATCH --time=02:00:00           # Set max runtime
#SBATCH --mail-type=ALL           # Notify user of job events
#SBATCH --mail-user=USER@scu.edu  # Replace with your email

# Load necessary modules
module purge
module load cuda/12.1 cudnn        # Load CUDA and cuDNN
module load python/3.10            # Load Python module (adjust if needed)

# Ensure the script exits on failure
set -e  

# Step 1: Ensure the virtual environment exists by running setup_env.sh if necessary
if [ ! -d "env" ]; then
    echo "Virtual environment not found. Running setup_env.sh..."
    bash setup_env.sh
fi

# Step 2: Activate the virtual environment
echo "Activating virtual environment..."
source env/bin/activate

# Step 3: Verify Python is using the virtual environment
echo "Using Python from: $(which python)"

# Step 4: Start the training script
echo "Starting training..."
python train_model3.py

echo "Training complete!"
