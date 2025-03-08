"""
TieredImageNet Dataset Setup Script

This script creates a structured dataset from TieredImageNet splits for use with the
"Making Better Mistakes" hierarchical classification framework. It extracts dataset
splits from zip files and organizes images into class-specific subfolders.

Usage:
    python setup_tiered_imagenet.py [--source-dir PATH] [--dest-dir PATH] [--splits-dir PATH]

Arguments:
    --source-dir: Path to source images directory (default: 'imagenet-data/imagenet-224/Data/CLS-LOC')
    --dest-dir: Path to destination directory for processed dataset (default: 'imagenet-data/tiered-imagenet-224')
    --splits-dir: Path to directory containing dataset splits ZIP files (default: 'dataset_splits')
    --dataset: Dataset to process, either 'tiered' for TieredImageNet or 'inat19' for iNaturalist19 (default: 'tiered')
    --train-subfolder: Whether train images are in class subfolders (default: auto-detect)
    --val-subfolder: Whether validation images are in class subfolders (default: auto-detect)  
    --test-subfolder: Whether test images are in class subfolders (default: auto-detect)
"""

import os
import zipfile
import shutil
import glob
import argparse
from pathlib import Path
import sys

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Setup TieredImageNet or iNaturalist19 dataset structure.')
    parser.add_argument('--source-dir', type=str, 
                        default='imagenet-data/imagenet-224/Data/CLS-LOC',
                        help='Path to source images directory')
    parser.add_argument('--dest-dir', type=str, 
                        default='imagenet-data/tiered-imagenet-224',
                        help='Path to destination directory for processed dataset')
    parser.add_argument('--splits-dir', type=str, 
                        default='dataset_splits',
                        help='Path to directory containing dataset splits ZIP files')
    parser.add_argument('--dataset', type=str, 
                        default='tiered', choices=['tiered', 'inat19'],
                        help='Dataset to process: tiered (TieredImageNet) or inat19 (iNaturalist19)')
    parser.add_argument('--train-subfolder', type=str,
                        default='auto', choices=['auto', 'yes', 'no'],
                        help='Whether train images are in class subfolders (auto, yes, no)')
    parser.add_argument('--val-subfolder', type=str,
                        default='auto', choices=['auto', 'yes', 'no'],
                        help='Whether validation images are in class subfolders (auto, yes, no)')
    parser.add_argument('--test-subfolder', type=str,
                        default='auto', choices=['auto', 'yes', 'no'],
                        help='Whether test images are in class subfolders (auto, yes, no)')
    return parser.parse_args()

def extract_splits(splits_dir, dataset):
    """
    Extract the dataset splits from zip file.
    
    Args:
        splits_dir: Directory containing the splits ZIP files
        dataset: Dataset type ('tiered' or 'inat19')
    
    Returns:
        Path to the extracted splits
    """
    # Define file paths based on dataset type
    if dataset == 'tiered':
        zip_file = os.path.join(splits_dir, 'splits_tiered.zip')
        extract_dir = 'temp_splits/tiered'
        splits_path = os.path.join(extract_dir, 'splits_tieredImageNet-H')
    else:  # inat19
        zip_file = os.path.join(splits_dir, 'splits_inat19.zip')
        extract_dir = 'temp_splits/inat19'
        splits_path = os.path.join(extract_dir, 'splits_iNaturalist19-H')
    
    print(f"Extracting {dataset} splits from {zip_file}...")
    
    # Check if the zip file exists
    if not os.path.exists(zip_file):
        print(f"Error: Splits file not found at {zip_file}")
        sys.exit(1)
        
    # Extract the zip file
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    print(f"Extraction completed to {extract_dir}")
    
    # Verify the extraction was successful
    if not os.path.exists(splits_path):
        print(f"Error: Expected splits not found at {splits_path} after extraction")
        sys.exit(1)
        
    return splits_path

def detect_folder_structure(source_dir, class_name, split_name):
    """
    Detect if images for a class are in a subfolder or directly in the split folder.
    
    Args:
        source_dir: Path to source images directory for the split
        class_name: Name of the class to check
        split_name: Name of the split ('train', 'val', 'test')
        
    Returns:
        Boolean indicating if images are in a subfolder
    """
    # Check if class subfolder exists
    class_dir = os.path.join(source_dir, class_name)
    if os.path.isdir(class_dir):
        # Check if there are any images in this folder (simple check for some image extensions)
        for ext in ['.jpg', '.jpeg', '.png']:
            if glob.glob(os.path.join(class_dir, f'*{ext}')):
                print(f"Detected subfolder structure for {split_name} split: images are in class-specific folders")
                return True
    
    # Check if images are directly in the split folder with class name as part of filename
    for ext in ['.jpg', '.jpeg', '.png']:
        if glob.glob(os.path.join(source_dir, f'{class_name}*{ext}')):
            print(f"Detected flat structure for {split_name} split: images are directly in the split folder")
            return False
            
    # Default case - assume no subfolders for val and test, yes for train
    if split_name == 'train':
        return True
    else:
        return False

def find_image_path(source_dir, class_name, image_name, use_subfolder, split_name):
    """
    Find the correct path to an image based on the folder structure.
    
    Args:
        source_dir: Base directory for the split
        class_name: Name of the class
        image_name: Name of the image file
        use_subfolder: Whether to look in class-specific subfolder
        split_name: Name of the split for logging
        
    Returns:
        Path to the image file
    """
    # Check if the image name already includes the class name prefix
    if image_name.startswith(class_name) and not use_subfolder:
        return os.path.join(source_dir, image_name)
    
    # Try with subfolder structure
    if use_subfolder:
        path = os.path.join(source_dir, class_name, image_name)
        if os.path.exists(path):
            return path
            
    # Try without subfolder
    path = os.path.join(source_dir, image_name)
    if os.path.exists(path):
        return path
    
    # Try with common prefixes or patterns
    for pattern in [f"{class_name}_{image_name}", f"{class_name}/{image_name}"]:
        path = os.path.join(source_dir, pattern)
        if os.path.exists(path):
            return path
    
    # If none of the above worked, return the most likely path based on the setting
    if use_subfolder:
        return os.path.join(source_dir, class_name, image_name)
    else:
        return os.path.join(source_dir, image_name)

def create_dataset_structure(source_base, dest_base, splits_path, dataset, subfolder_settings):
    """
    Create the dataset directory structure.
    
    Args:
        source_base: Path to source images directory
        dest_base: Path to destination directory for processed dataset
        splits_path: Path to the extracted splits directory
        dataset: Dataset type ('tiered' or 'inat19')
        subfolder_settings: Dictionary with split names as keys and subfolder settings as values
    """
    print(f"Creating {dataset} dataset directory structure...")
    
    # Print paths for verification
    print(f"Source images directory: {source_base}")
    print(f"Destination directory: {dest_base}")
    print(f"Splits directory: {splits_path}")
    
    # Create destination directory
    os.makedirs(dest_base, exist_ok=True)
    
    # Define splits to process
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_source_dir = os.path.join(splits_path, split)
        if not os.path.exists(split_source_dir):
            print(f"Warning: {split_source_dir} doesn't exist, skipping...")
            continue
            
        print(f"Processing {split} split...")
        
        # Create output directory for this split
        split_output_dir = os.path.join(dest_base, 'Data', 'CLS-LOC', split)
        os.makedirs(split_output_dir, exist_ok=True)
        
        # Source images directory for this split
        source_images_dir = os.path.join(source_base, split)
        if not os.path.exists(source_images_dir):
            print(f"Warning: Source directory {source_images_dir} doesn't exist!")
            continue
        
        # Get all class txt files in this split
        class_files = glob.glob(os.path.join(split_source_dir, "*.txt"))
        total_classes = len(class_files)
        
        print(f"Found {total_classes} classes in {split} split")
        
        # Determine folder structure if set to auto
        use_subfolder = subfolder_settings[split]
        if use_subfolder == 'auto' and total_classes > 0:
            # Get first class and check its structure
            first_class = os.path.basename(class_files[0]).split('.')[0]
            use_subfolder = detect_folder_structure(source_images_dir, first_class, split)
        elif use_subfolder == 'yes':
            use_subfolder = True
        else:
            use_subfolder = False
            
        print(f"Using {'subfolder' if use_subfolder else 'flat'} structure for {split} split")
        
        for class_idx, class_file in enumerate(class_files, 1):
            # Get class name from filename (without extension)
            class_name = os.path.basename(class_file).split('.')[0]
            
            # Create class directory in the output
            class_output_dir = os.path.join(split_output_dir, class_name)
            os.makedirs(class_output_dir, exist_ok=True)
            
            print(f"Processing class: {class_name} ({class_idx}/{total_classes})")
            
            # Read image names from the class file
            with open(class_file, 'r') as f:
                image_names = [line.strip() for line in f.readlines()]
            
            num_images = len(image_names)
            print(f"  - Found {num_images} images")
            copied_images = 0
            skipped_images = 0
            
            # Copy each image to the class directory
            for image_idx, image_name in enumerate(image_names):
                source_image_path = find_image_path(source_images_dir, class_name, image_name, use_subfolder, split)
                dest_image_path = os.path.join(class_output_dir, os.path.basename(image_name))
                
                # Check if source image exists
                if os.path.exists(source_image_path):
                    try:
                        shutil.copy2(source_image_path, dest_image_path)
                        copied_images += 1
                        # Show progress every 100 images or at the end
                        if image_idx % 100 == 0 or image_idx == num_images - 1:
                            print(f"    Progress: {image_idx + 1}/{num_images} images processed", end='\r')
                    except Exception as e:
                        print(f"\nError copying {source_image_path}: {e}")
                        skipped_images += 1
                else:
                    skipped_images += 1
                    if image_idx < 5 or image_idx % 1000 == 0:  # Only show first few errors or occasionally to avoid flooding
                        print(f"\nWarning: Source image not found: {source_image_path}")
            
            print(f"\n  - Successfully copied {copied_images}/{num_images} images for class {class_name} (Skipped: {skipped_images})")
    
    print("\nDataset directory structure created successfully.")
    print(f"Dataset is organized at: {os.path.join(dest_base, 'Data', 'CLS-LOC')}")

def main():
    """Main function to set up the dataset."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up subfolder settings based on arguments
    subfolder_settings = {
        'train': args.train_subfolder,
        'val': args.val_subfolder,
        'test': args.test_subfolder
    }
    
    # Extract the dataset splits
    splits_path = extract_splits(args.splits_dir, args.dataset)
    
    # Create the dataset directory structure
    create_dataset_structure(args.source_dir, args.dest_dir, splits_path, args.dataset, subfolder_settings)
    
    print("Setup completed. Please review the directory structure and make any necessary adjustments.")
    print("\nTo use the dataset with the training script, run:")
    print(f"python scripts/start_training.py --arch resnet18 --loss cross-entropy --lr 1e-5 --data {args.dataset}-224 --workers 8 --data-paths-config data_paths.yml --output output_folder/ --num_training_steps 200000")

if __name__ == "__main__":
    main()