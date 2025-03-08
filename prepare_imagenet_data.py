#!/usr/bin/env python
"""
ImageNet Data Preparation Script for the "Making Better Mistakes" Framework

This script automates the entire process of preparing ImageNet data:
1. Extracts the ImageNet object localization challenge zip file
2. Detects the folder structure of the extracted data
3. Resizes all images to the dimensions required by the CNN (224x224)
4. Creates the dataset splits according to TieredImageNet requirements

Usage:
    python prepare_imagenet_data.py [options]

Required Arguments:
    --zip-file PATH       Path to the imagenet-object-localization-challenge.zip file

Optional Arguments:
    --output-dir PATH     Path where extracted and processed data should be saved
                         (default: imagenet-data/)
    --resize-target SIZE  Target size for resizing images (default: 224x224!)
    --cpu-workers INT     Number of CPU workers for parallel processing (default: auto)
    --skip-extract        Skip extraction if data is already extracted
    --skip-resize         Skip resize if resized data already exists
    --skip-splits         Skip creating splits if they already exist
    --dataset             Dataset to process ('tiered' or 'inat19') (default: tiered)
"""
import os
import sys
import logging
import argparse
import zipfile
import subprocess
import traceback
import shutil
import glob
import yaml
from pathlib import Path
from PIL import Image, ImageFile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('imagenet_prep')

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Import required functions from existing scripts
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.resize import resize_image, parse_size_string, find_image_files, process_batch
from setup_tiered_imagenet import extract_splits, detect_folder_structure, find_image_path, create_dataset_structure

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Prepare ImageNet data for the "Making Better Mistakes" framework.')
    parser.add_argument('--zip-file', type=str, required=True,
                        help='Path to the imagenet-object-localization-challenge.zip file')
    parser.add_argument('--output-dir', type=str, default='imagenet-data',
                        help='Path where extracted and processed data should be saved')
    parser.add_argument('--resize-target', type=str, default='224x224!',
                        help='Target size for resizing images (e.g., "224x224!", "299x299!", etc.)')
    parser.add_argument('--cpu-workers', type=int, default=os.cpu_count(),
                        help=f'Number of CPU workers for parallel processing (default: {os.cpu_count()})')
    parser.add_argument('--skip-extract', action='store_true',
                        help='Skip extraction if data is already extracted')
    parser.add_argument('--skip-resize', action='store_true',
                        help='Skip resize if resized data already exists')
    parser.add_argument('--skip-splits', action='store_true',
                        help='Skip creating splits if they already exist')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing files')
    parser.add_argument('--dataset', type=str, default='tiered', choices=['tiered', 'inat19'],
                        help='Dataset to process: tiered (TieredImageNet) or inat19 (iNaturalist19)')
    parser.add_argument('--inat-source', type=str, default='inaturalist-data/train_val',
                        help='Source directory for iNaturalist19 images (only used when dataset=inat19)')
    parser.add_argument('--json-file', type=str, default=None,
                        help='Path to iNaturalist19 JSON file (only used when dataset=inat19)')
    return parser.parse_args()

def extract_imagenet_zip(zip_file, output_dir, skip_extract=False):
    """
    Extract the ImageNet zip file to the output directory.
    
    Args:
        zip_file: Path to the ImageNet zip file
        output_dir: Directory to extract the contents to
        skip_extract: Skip extraction if data is already extracted
        
    Returns:
        Path to the extracted data directory
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create paths
    extracted_dir = os.path.join(output_dir, 'ILSVRC')
    
    # Check if already extracted
    if os.path.exists(extracted_dir) and skip_extract:
        logger.info(f"Skipping extraction as {extracted_dir} already exists and skip-extract is enabled")
        return extracted_dir
    
    # Check if the zip file exists
    if not os.path.exists(zip_file):
        logger.error(f"ImageNet zip file not found: {zip_file}")
        sys.exit(1)
    
    logger.info(f"Extracting {zip_file} to {output_dir}...")
    
    # Get the total size of the zip for progress tracking
    zip_size = os.path.getsize(zip_file)
    logger.info(f"Zip file size: {zip_size / (1024 * 1024 * 1024):.2f} GB")
    
    try:
        # First check if 7-Zip is available (much faster for large archives)
        try:
            # Try to run 7z command to check if it's installed
            subprocess.check_output(["7z", "--help"], stderr=subprocess.STDOUT)
            logger.info("Using 7-Zip for extraction (faster)")
            
            # Extract using 7-Zip
            cmd = ["7z", "x", zip_file, f"-o{output_dir}", "-aoa"]
            subprocess.check_call(cmd)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fall back to Python's built-in zipfile if 7-Zip not available
            logger.info("7-Zip not found, using Python's zipfile module (slower)")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Get total number of files for progress tracking
                total_files = len(zip_ref.namelist())
                logger.info(f"Total files in archive: {total_files}")
                
                # Extract with progress updates
                extracted_files = 0
                for file in zip_ref.namelist():
                    zip_ref.extract(file, output_dir)
                    extracted_files += 1
                    
                    # Show progress every 1000 files
                    if extracted_files % 1000 == 0:
                        logger.info(f"Extracted {extracted_files}/{total_files} files ({extracted_files/total_files*100:.1f}%)")
    except Exception as e:
        logger.error(f"Error during extraction: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    # Verify extraction
    if not os.path.exists(extracted_dir):
        logger.error(f"Extraction failed: {extracted_dir} not found")
        sys.exit(1)
    
    logger.info(f"Successfully extracted ImageNet zip to {extracted_dir}")
    return extracted_dir

def detect_imagenet_structure(extracted_dir):
    """
    Detect the structure of the extracted ImageNet data.
    
    Args:
        extracted_dir: Directory containing the extracted ImageNet data
        
    Returns:
        Dictionary with information about the data structure
    """
    logger.info(f"Detecting ImageNet data structure in {extracted_dir}...")
    
    structure = {
        'train_dir': None,
        'val_dir': None,
        'test_dir': None,
        'has_subfolders': {},
        'class_count': {},
        'image_count': {}
    }
    
    # Common patterns for ImageNet directories
    patterns = [
        {'train': 'train', 'val': 'val', 'test': 'test'},
        {'train': 'train', 'val': 'validation', 'test': 'test'},
        {'train': 'Data/CLS-LOC/train', 'val': 'Data/CLS-LOC/val', 'test': 'Data/CLS-LOC/test'}
    ]
    
    # Try each pattern to find the correct structure
    for pattern in patterns:
        found = True
        for split, path in pattern.items():
            full_path = os.path.join(extracted_dir, path)
            if not os.path.exists(full_path):
                found = False
                break
            
            # Store the path if it exists
            if found:
                structure[f'{split}_dir'] = full_path
                
        if found:
            logger.info(f"Found ImageNet folder structure matching pattern: {pattern}")
            break
    
    # If no structure was found
    if structure['train_dir'] is None:
        # Search for common folder names
        for dirname in os.listdir(extracted_dir):
            full_path = os.path.join(extracted_dir, dirname)
            if not os.path.isdir(full_path):
                continue
                
            if 'train' in dirname.lower():
                structure['train_dir'] = full_path
            elif 'val' in dirname.lower():
                structure['val_dir'] = full_path
            elif 'test' in dirname.lower():
                structure['test_dir'] = full_path
    
    # Check if we found the directories
    if structure['train_dir'] is None:
        logger.error("Could not detect ImageNet directory structure. Please organize data manually.")
        sys.exit(1)
        
    logger.info(f"Detected train directory: {structure['train_dir']}")
    logger.info(f"Detected val directory: {structure['val_dir']}")
    logger.info(f"Detected test directory: {structure['test_dir']}")
    
    # Analyze each directory to detect subfolders and count images
    for split_name in ['train', 'val', 'test']:
        split_dir = structure[f'{split_name}_dir']
        if not split_dir:
            continue
            
        # Check if images are in class subfolders
        subfolders = [f for f in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, f))]
        has_subfolders = len(subfolders) > 0
        structure['has_subfolders'][split_name] = has_subfolders
        
        # Count classes and images
        if has_subfolders:
            structure['class_count'][split_name] = len(subfolders)
            total_images = sum(len(glob.glob(os.path.join(split_dir, subfolder, '*.[jJpP][pPnN][gG]'))) for subfolder in subfolders)
            structure['image_count'][split_name] = total_images
            logger.info(f"{split_name} split: {len(subfolders)} classes, {total_images} images (in subfolders)")
        else:
            total_images = len(glob.glob(os.path.join(split_dir, '*.[jJpP][pPnN][gG]')))
            structure['image_count'][split_name] = total_images
            logger.info(f"{split_name} split: {total_images} images (flat structure)")
    
    return structure

def resize_imagenet_data(structure, output_dir, resize_target, cpu_workers, skip_resize=False, overwrite=False):
    """
    Resize the ImageNet images to the target size.
    
    Args:
        structure: Dictionary with information about the data structure
        output_dir: Base output directory
        resize_target: Target size for resizing (e.g., '224x224!')
        cpu_workers: Number of CPU workers for parallel processing
        skip_resize: Skip resize if output already exists
        overwrite: Overwrite existing files
        
    Returns:
        Dictionary with paths to resized data
    """
    # Create output paths
    resized_dir = os.path.join(output_dir, f'imagenet-{resize_target.split("x")[0]}')
    
    # Create the resize mode dictionary
    width, height, resize_mode = parse_size_string(resize_target)
    
    resized_structure = {
        'base_dir': resized_dir,
        'train_dir': None,
        'val_dir': None,
        'test_dir': None
    }
    
    # Process each split
    for split_name in ['train', 'val', 'test']:
        split_dir = structure.get(f'{split_name}_dir')
        if not split_dir or not os.path.exists(split_dir):
            logger.info(f"Skipping {split_name} split: directory not found")
            continue
            
        # Define output directory for this split
        output_split_dir = os.path.join(resized_dir, 'Data', 'CLS-LOC', split_name)
        
        # Store the output path in the structure
        resized_structure[f'{split_name}_dir'] = output_split_dir
        
        # Check if we can skip
        if os.path.exists(output_split_dir) and skip_resize:
            logger.info(f"Skipping resize for {split_name} split: output directory already exists")
            continue
            
        # Create the output directory
        os.makedirs(output_split_dir, exist_ok=True)
        
        logger.info(f"Resizing {split_name} split images to {width}x{height}...")
        
        # Check if we have class subfolders
        has_subfolders = structure['has_subfolders'].get(split_name, False)
        
        if has_subfolders:
            # Process each class subfolder
            class_folders = [f for f in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, f))]
            logger.info(f"Found {len(class_folders)} class folders to process")
            
            for class_idx, class_folder in enumerate(class_folders, 1):
                logger.info(f"Processing class {class_idx}/{len(class_folders)}: {class_folder}")
                
                # Create output directory for this class
                output_class_dir = os.path.join(output_split_dir, class_folder)
                os.makedirs(output_class_dir, exist_ok=True)
                
                # Find all images for this class
                class_dir = os.path.join(split_dir, class_folder)
                class_images = find_image_files(class_dir, ['.jpg', '.jpeg', '.png'])
                
                # Convert relative paths to absolute paths
                class_images = [os.path.relpath(os.path.join(class_dir, img), class_dir) for img in class_images]
                
                if not class_images:
                    logger.warning(f"No images found in {class_dir}")
                    continue
                    
                logger.info(f"  - Found {len(class_images)} images to resize")
                
                # Create batches for parallel processing
                batch_size = max(10, len(class_images) // cpu_workers)
                batches = [class_images[i:i+batch_size] for i in range(0, len(class_images), batch_size)]
                
                # Process batches in parallel
                results = []
                for batch in batches:
                    result = process_batch(batch, class_dir, output_class_dir, width, height, resize_mode, 95, overwrite)
                    results.append(result)
                
                # Combine results
                total_success = sum(r['success'] for r in results)
                total_failed = sum(r['failed'] for r in results)
                total_skipped = sum(r['skipped'] for r in results)
                
                logger.info(f"  - Resized {total_success} images (Failed: {total_failed}, Skipped: {total_skipped})")
        else:
            # Process all images in the flat structure
            logger.info(f"Processing images in flat structure")
            
            # Find all images in this split
            split_images = find_image_files(split_dir, ['.jpg', '.jpeg', '.png'])
            
            # Convert relative paths to absolute paths
            split_images = [os.path.relpath(os.path.join(split_dir, img), split_dir) for img in split_images]
            
            if not split_images:
                logger.warning(f"No images found in {split_dir}")
                continue
                
            logger.info(f"Found {len(split_images)} images to resize")
            
            # Create batches for parallel processing
            batch_size = max(10, len(split_images) // cpu_workers)
            batches = [split_images[i:i+batch_size] for i in range(0, len(split_images), batch_size)]
            
            # Process batches in parallel
            results = []
            for batch in batches:
                result = process_batch(batch, split_dir, output_split_dir, width, height, resize_mode, 95, overwrite)
                results.append(result)
            
            # Combine results
            total_success = sum(r['success'] for r in results)
            total_failed = sum(r['failed'] for r in results)
            total_skipped = sum(r['skipped'] for r in results)
            
            logger.info(f"Resized {total_success} images (Failed: {total_failed}, Skipped: {total_skipped})")
    
    logger.info(f"Image resizing completed. Resized images are in {resized_dir}")
    
    return resized_structure

def organize_validation_data(val_dir):
    """
    Organize validation data into class folders using the validation labels.
    
    Args:
        val_dir: Path to the validation directory
    """
    try:
        # Check if the directory has class subfolders
        subfolders = [f for f in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, f))]
        if len(subfolders) > 0:
            logger.info("Validation data already organized into class folders")
            return
            
        logger.info("Organizing validation data into class folders...")
        
        # Look for validation labels file
        val_label_file = os.path.join(val_dir, 'val_annotations.txt')
        if not os.path.exists(val_label_file):
            val_label_file = os.path.join(os.path.dirname(val_dir), 'val_annotations.txt')
        
        if not os.path.exists(val_label_file):
            logger.warning("Validation label file not found. Cannot organize validation data.")
            return
            
        # Read the validation labels
        with open(val_label_file, 'r') as f:
            val_labels = [line.strip().split() for line in f.readlines()]
            
        # Create class folders and move images
        for image_name, class_id, *_ in val_labels:
            # Create class folder if it doesn't exist
            class_dir = os.path.join(val_dir, class_id)
            os.makedirs(class_dir, exist_ok=True)
            
            # Move image to class folder
            src_path = os.path.join(val_dir, image_name)
            dst_path = os.path.join(class_dir, image_name)
            
            if os.path.exists(src_path):
                shutil.move(src_path, dst_path)
                
        logger.info("Validation data organized into class folders successfully")
        
    except Exception as e:
        logger.error(f"Error organizing validation data: {e}")
        logger.error(traceback.format_exc())

def create_dataset_splits_and_config(resized_structure, output_dir, dataset, skip_splits=False):
    """
    Create dataset splits according to TieredImageNet or iNaturalist19 requirements.
    
    Args:
        resized_structure: Dictionary with paths to resized data
        output_dir: Base output directory
        dataset: Dataset to process ('tiered' or 'inat19')
        skip_splits: Skip creating splits if they already exist
        
    Returns:
        Path to the created dataset directory
    """
    logger.info(f"Creating {dataset} splits...")
    
    # Define paths based on dataset
    if dataset == 'tiered':
        dataset_dir = os.path.join(output_dir, 'tiered-imagenet-224')
        dataset_key = 'tiered-imagenet-224'
    else:  # inat19
        dataset_dir = os.path.join(output_dir, 'inaturalist19-224')
        dataset_key = 'inaturalist19-224'
    
    # Check if splits already exist
    if skip_splits and os.path.exists(dataset_dir):
        logger.info(f"Skipping split creation: {dataset_dir} already exists")
        return dataset_dir
    
    # Extract the dataset splits from the provided zip files
    splits_dir = os.path.join(output_dir, '..', 'dataset_splits')
    
    if dataset == 'tiered':
        if not os.path.exists(os.path.join(splits_dir, 'splits_tiered.zip')):
            logger.error(f"Splits file not found: {os.path.join(splits_dir, 'splits_tiered.zip')}")
            sys.exit(1)
    else:  # inat19
        if not os.path.exists(os.path.join(splits_dir, 'splits_inat19.zip')):
            logger.error(f"Splits file not found: {os.path.join(splits_dir, 'splits_inat19.zip')}")
            sys.exit(1)
    
    # Extract the splits
    splits_path = extract_splits(splits_dir, dataset)
    
    # Set up subfolder settings - we've already organized the data correctly
    subfolder_settings = {
        'train': 'yes',
        'val': 'yes',
        'test': 'yes'
    }
    
    # Create the dataset structure using the existing function
    create_dataset_structure(
        resized_structure['base_dir'] + '/Data/CLS-LOC',
        dataset_dir,
        splits_path, 
        dataset,
        subfolder_settings
    )
    
    # Update data_paths.yml with the new paths
    data_paths_file = os.path.join(output_dir, '..', 'data_paths.yml')
    data_paths_example = os.path.join(output_dir, '..', 'data_paths.yml.example')
    
    # If data_paths.yml doesn't exist but example does, create from example
    if not os.path.exists(data_paths_file) and os.path.exists(data_paths_example):
        shutil.copy2(data_paths_example, data_paths_file)
        logger.info(f"Created {data_paths_file} from {data_paths_example}")
    
    # Load existing data paths if available
    data_paths = {}
    if os.path.exists(data_paths_file):
        try:
            with open(data_paths_file, 'r') as f:
                data_paths = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Error reading {data_paths_file}: {e}")
    
    # Update paths
    dataset_path = os.path.abspath(os.path.join(dataset_dir, 'Data', 'CLS-LOC'))
    data_paths[dataset_key] = dataset_path
    
    # Write updated data_paths.yml
    try:
        with open(data_paths_file, 'w') as f:
            yaml.dump(data_paths, f, default_flow_style=False)
        logger.info(f"Updated {data_paths_file} with path: {dataset_path}")
    except Exception as e:
        logger.error(f"Error writing to {data_paths_file}: {e}")
    
    return dataset_dir

def prepare_inaturalist_dataset(args):
    """
    Prepare the iNaturalist19 dataset using the dedicated script.
    
    Args:
        args: Command line arguments
        
    Returns:
        Path to the created iNaturalist19 dataset directory
    """
    logger.info("Preparing iNaturalist19 dataset...")
    
    # Import the setup_inaturalist19 module
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from setup_inaturalist19 import extract_splits, create_dataset_structure, load_category_mapping, update_data_paths_yml
    except ImportError:
        logger.error("Could not import setup_inaturalist19 module. Make sure the file exists.")
        sys.exit(1)
    
    # Define paths
    source_dir = args.inat_source
    dest_dir = os.path.join(args.output_dir, 'inaturalist19-224')
    splits_dir = os.path.join(args.output_dir, '..', 'dataset_splits')
    json_file = args.json_file
    
    # Set up subfolder settings
    subfolder_settings = {
        'train': 'auto',
        'val': 'auto',
        'test': 'auto'
    }
    
    # Load category mapping if JSON file is provided
    if not json_file:
        # Try to find JSON file in common locations
        common_locations = [
            os.path.join(source_dir, 'train2019.json'),
            os.path.join(source_dir, '..', 'train2019.json'),
            'inaturalist-data/train2019.json',
            'train2019.json'
        ]
        for location in common_locations:
            if os.path.exists(location):
                json_file = location
                logger.info(f"Found iNaturalist JSON file at: {json_file}")
                break
    
    category_to_dir = load_category_mapping(json_file)
    
    # Extract the dataset splits
    splits_path = extract_splits(splits_dir)
    
    # Create the dataset structure
    create_dataset_structure(
        source_dir,
        dest_dir,
        splits_path,
        subfolder_settings,
        category_to_dir,
        args.resize_target if hasattr(args, 'resize_target') else None
    )
    
    # Update data_paths.yml
    update_data_paths_yml(dest_dir)
    
    return dest_dir

def main():
    """Main function to prepare ImageNet data."""
    args = parse_arguments()
    
    try:
        # Handle iNaturalist19 dataset directly if selected
        if args.dataset == 'inat19':
            inat_dir = prepare_inaturalist_dataset(args)
            logger.info("\nData preparation completed successfully!")
            logger.info(f"iNaturalist19 dataset is ready at: {inat_dir}")
            logger.info("\nTo run experiments, use commands like:")
            logger.info("cd experiments")
            logger.info("bash crossentropy_inaturalist19.sh")
            return
        
        # For TieredImageNet, follow the full process:
        # 1. Extract the ImageNet zip file
        extracted_dir = extract_imagenet_zip(
            args.zip_file, 
            args.output_dir,
            args.skip_extract
        )
        
        # 2. Detect the folder structure of the extracted data
        structure = detect_imagenet_structure(extracted_dir)
        
        # 3. Resize all images to the required dimensions
        resized_structure = resize_imagenet_data(
            structure,
            args.output_dir,
            args.resize_target,
            args.cpu_workers,
            args.skip_resize,
            args.overwrite
        )
        
        # 4. Create the dataset splits according to TieredImageNet requirements
        tiered_dir = create_dataset_splits_and_config(
            resized_structure,
            args.output_dir,
            args.dataset,
            args.skip_splits
        )
        
        logger.info("\nData preparation completed successfully!")
        logger.info(f"TieredImageNet dataset is ready at: {tiered_dir}")
        logger.info("\nTo run experiments, use commands like:")
        logger.info("cd experiments")
        logger.info("bash crossentropy_tieredimagenet.sh")
        
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}")
        logger.critical(traceback.format_exc())
        sys.exit(1)
    
if __name__ == "__main__":
    main()