#!/bin/bash

# Script to run all inaturalist19 experiments sequentially
# Created on March 13, 2025

echo "Starting all inaturalist19 experiments"
echo "======================================="

# Function to run experiment and log result
run_experiment() {
    echo "Starting experiment: $1"
    echo "Time: $(date)"
    echo "----------------------------------------"
    
    # Run the experiment
    bash "$1"
    
    # Check if experiment completed successfully
    if [ $? -eq 0 ]; then
        echo "Experiment $1 completed successfully"
    else
        echo "Experiment $1 failed with exit code $?"
    fi
    
    echo "----------------------------------------"
    echo "Finished experiment: $1"
    echo "Time: $(date)"
    echo ""
}

# Run all inaturalist19 experiments
run_experiment "experiments/barzdenzler_inaturalist19.sh"
run_experiment "experiments/crossentropy_inaturalist19.sh"
run_experiment "experiments/hxe_inaturalist19_alpha0.5.sh"
run_experiment "experiments/softlabels_inaturalist19_beta15.sh"
run_experiment "experiments/yolov2_inaturalist19.sh"

echo "All inaturalist19 experiments completed"
echo "Time: $(date)"
