#!/bin/bash

# Setup script for making-better-mistakes experiments
# This script sets up everything needed to run the experiments

set -e  # Exit on error

echo "==============================================================="
echo "Setting up environment for making-better-mistakes experiments"
echo "==============================================================="

# 1. Create/update conda environment
echo -e "\n[1/3] Setting up conda environment..."
if command -v conda &> /dev/null; then
    echo "Conda found, updating/creating environment from environment.yml"
    conda env update -f environment.yml
    pip install -e .
    echo "Conda environment 'better_mistakes' created/updated successfully"
    echo "Activate it with: conda activate better_mistakes"
else
    echo "Error: conda not found. Please install miniconda or anaconda first."
    exit 1
fi

# 2. Create data_paths.yml if it doesn't exist
echo -e "\n[2/3] Setting up data paths configuration..."
if [ ! -f data_paths.yml ]; then
    echo "Creating data_paths.yml from example file"
    cp data_paths.yml.example data_paths.yml
    echo "Created data_paths.yml - please edit this file to point to your dataset locations"
else
    echo "data_paths.yml already exists, please ensure it points to the correct dataset locations"
fi

# 3. Make experiment scripts executable
echo -e "\n[3/3] Making experiment scripts executable..."
chmod +x experiments/*.sh
echo "All experiment scripts are now executable"

# Final instructions
echo -e "\n==============================================================="
echo "Setup complete! Next steps:"
echo "==============================================================="
echo "1. Activate the conda environment:     conda activate better_mistakes"
echo "2. Edit data_paths.yml:                nano data_paths.yml"
echo "3. Setup the datasets if needed (e.g., download iNaturalist19)"
echo "4. Run experiments individually:       ./experiments/barzdenzler_inaturalist19.sh"
echo "5. Or run all iNaturalist19 experiments sequentially: ./experiments/run_all_inaturalist19.sh"
echo ""
echo "Note: Before running experiments, make sure:"
echo "- The conda environment is activated"
echo "- data_paths.yml points to your dataset locations"
echo "- You have sufficient GPU resources available"
echo "==============================================================="
