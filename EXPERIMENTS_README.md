# Making Better Mistakes - Experiments Guide

This document provides detailed instructions for setting up and running the experiments from the paper:

**[Making Better Mistakes: Leveraging Class Hierarchies with Deep Networks](https://arxiv.org/abs/1912.09393)**  
Luca Bertinetto*, Romain Mueller*, Konstantinos Tertikas, Sina Samangooei, Nicholas A. Lord*.  
_IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2020_

## Quick Start

```bash
# 1. Setup the environment and prepare datasets
./setup_environment.sh

# 2. Activate the conda environment
conda activate better_mistakes

# 3. Run all iNaturalist19 experiments sequentially
./experiments/run_all_inaturalist19.sh
```

## Detailed Setup Instructions

### 1. Environment Setup

The `setup_environment.sh` script handles most of the setup process:

```bash
./setup_environment.sh
```

This script performs the following steps:
- Creates/updates the conda environment from `environment.yml`
- Sets up data paths configuration
- Verifies dataset preparation scripts are available
- Pulls Git LFS files if necessary
- Makes all experiment scripts executable

### 2. Dataset Preparation

The iNaturalist19 dataset needs to be prepared before running experiments:

```bash
# Download iNaturalist19 dataset (if you haven't already)
# Visit https://github.com/visipedia/inat_comp for instructions

# Run the setup script, providing paths to your data
python setup_inaturalist19.py --source-dir /path/to/inaturalist-data --dest-dir /path/for/processed-data
```

After dataset preparation, update `data_paths.yml` with the correct paths:

```yaml
inaturalist19-224: '/path/to/your/processed/iNaturalist19/data'
```

### 3. Running Experiments

#### Run All iNaturalist19 Experiments

The `run_all_inaturalist19.sh` script sequentially executes all iNaturalist19 experiments:

```bash
./experiments/run_all_inaturalist19.sh
```

This will run the following experiments in sequence:
- barzdenzler_inaturalist19.sh
- crossentropy_inaturalist19.sh
- hxe_inaturalist19_alpha*.sh (with various alpha values)
- softlabels_inaturalist19_beta*.sh (with various beta values)
- yolov2_inaturalist19.sh

Each experiment logs its progress, timing information, and success/failure status.

#### Run Individual Experiments

You can also run any experiment individually:

```bash
./experiments/crossentropy_inaturalist19.sh
```

## Experiment Methods

The repository includes implementations of different methods for hierarchical classification:

1. **Cross-Entropy**: Standard cross-entropy loss (baseline)
2. **Soft Labels**: Hierarchical soft labels approach
3. **HXE (Hierarchical Cross-Entropy)**: Hierarchical cross-entropy with different alpha parameters
4. **Barz & Denzler**: Implementation of the method from Barz & Denzler (2019)
5. **YOLOv2**: Hierarchical loss function from YOLOv2

## Results Analysis

After running experiments, you can analyze the results:

```bash
# Plot trade-offs between flat and hierarchical accuracy
python scripts/plot_tradeoffs.py --results-dir /path/to/results

# Run tests on the best models determined by plot_tradeoffs.py
python scripts/start_testing.py --results-dir /path/to/results
```

## Troubleshooting

- **Memory Issues**: If you encounter GPU memory issues, consider reducing the batch size in the experiment scripts
- **Dataset Paths**: Ensure your `data_paths.yml` has the correct absolute paths
- **Environment Activation**: Always ensure the conda environment is activated with `conda activate better_mistakes`
- **GPU Availability**: Verify GPU availability with `nvidia-smi` before running experiments
