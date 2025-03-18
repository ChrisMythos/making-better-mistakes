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
    bash $1
    
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
run_experiment barzdenzler_inaturalist19.sh
run_experiment crossentropy_inaturalist19.sh
run_experiment softlabels_inaturalist19_beta15.sh
run_experiment hxe_inaturalist19_alpha0.5.sh
run_experiment yolov2_inaturalist19.sh

echo "All diffrent inaturalist19 experiments are completed. Now running additional experiments with varying hyperparameters"
echo "Time: $(date)"

# Additional experiments for inaturalist19 with varying hyperparameters
run_experiment softlabels_inaturalist19_beta04.sh
run_experiment softlabels_inaturalist19_beta05.sh
run_experiment softlabels_inaturalist19_beta10.sh
run_experiment hxe_inaturalist19_alpha0.6.sh
run_experiment hxe_inaturalist19_alpha0.7.sh
run_experiment hxe_inaturalist19_alpha0.8.sh
run_experiment hxe_inaturalist19_alpha0.9.sh
run_experiment hxe_inaturalist19_alpha0.1.sh
run_experiment hxe_inaturalist19_alpha0.2.sh
run_experiment hxe_inaturalist19_alpha0.3.sh
run_experiment hxe_inaturalist19_alpha0.4.sh
run_experiment softlabels_inaturalist19_beta20.sh
run_experiment softlabels_inaturalist19_beta30.sh

echo "All inaturalist19 experiments completed"
echo "Time: $(date)"