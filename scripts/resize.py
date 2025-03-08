#!/usr/bin/env python
"""
Image Resizing Script

This script batch resizes images from a source directory to a destination directory
using various resizing options similar to ImageMagick's convert command.

Usage:
    python resize.py [options] src_directory dst_directory size_string

Size String Options:
    WxH     - Resize to WxH, preserving aspect ratio
    WxH!    - Force resize to exact dimensions
    WxH>    - Resize only if image is larger than WxH
    WxH^>   - Resize to fill the WxH area, may be larger than WxH

Examples:
    python resize.py images/ resized/ 224x224!     # Exact size
    python resize.py images/ resized/ 640x360>     # Shrink to fit
    python resize.py images/ resized/ 360x360^>    # Fill area

Additional Options:
    --include-exts      Additional file extensions to include
    --exclude-dirs      Directories to exclude from processing
    --copy-other-files  Copy non-image files to destination (default: False)
    --num-parallel      Number of parallel threads (default: CPU count)
    --quality           JPEG quality (0-100, default: 95)
    --overwrite         Overwrite existing files (default: False)
    --log-file          Path to log file
    --verbose           Print detailed progress information
"""

import os
import sys
import argparse
import glob
import concurrent.futures
from PIL import Image, ImageFile
import shutil
import logging
import time
from datetime import datetime
import traceback

# Allow truncated images to be processed
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('image_resizer')

def configure_logging(log_file=None, verbose=False):
    """Configure logging based on command line arguments."""
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

def parse_size_string(size_str):
    """Parse the size string and return width, height, and resize mode."""
    resize_mode = {
        'method': Image.LANCZOS,  # Default resampling method
        'exact': False,           # Force exact dimensions
        'shrink_only': False,     # Only shrink, never enlarge
        'fill': False             # Fill the target dimensions
    }
    
    # Parse size string with modifiers
    if size_str.endswith('!'):
        # Stretch to exact size
        width, height = map(int, size_str[:-1].split('x'))
        resize_mode['exact'] = True
    elif size_str.endswith('>'):
        if size_str.endswith('^>'):
            # Fill
            width, height = map(int, size_str[:-2].split('x'))
            resize_mode['fill'] = True
        else:
            # Fit (shrink only)
            width, height = map(int, size_str[:-1].split('x'))
            resize_mode['shrink_only'] = True
    else:
        # Default resize
        width, height = map(int, size_str.split('x'))
    
    return width, height, resize_mode

def resize_image(src_path, dst_path, width, height, resize_mode, quality=95, overwrite=False):
    """
    Resize an image using PIL with the given parameters.
    
    Args:
        src_path (str): Path to source image
        dst_path (str): Path to save resized image
        width (int): Target width
        height (int): Target height
        resize_mode (dict): Dict with resize options
        quality (int): JPEG quality (1-100)
        overwrite (bool): Whether to overwrite existing files
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Skip if destination exists and we're not overwriting
    if os.path.exists(dst_path) and not overwrite:
        logger.debug(f"Skipping {dst_path} (already exists)")
        return False
    
    try:
        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        
        # Open the image
        img = Image.open(src_path)
        
        # Convert to RGB if RGBA to avoid issues with some formats
        if img.mode == 'RGBA' and dst_path.lower().endswith(('.jpg', '.jpeg')):
            img = img.convert('RGB')
        
        # Get original dimensions
        orig_width, orig_height = img.size
        
        # Determine if we need to resize
        if resize_mode['shrink_only'] and orig_width <= width and orig_height <= height:
            # No need to resize, just copy
            img.save(dst_path, quality=quality, optimize=True)
            logger.debug(f"Copied without resizing: {src_path} -> {dst_path}")
            return True
        
        # Calculate new dimensions
        if resize_mode['fill']:
            # Fill the target area (similar to ImageMagick's ^ operator)
            ratio = max(width / orig_width, height / orig_height)
            new_width = int(orig_width * ratio)
            new_height = int(orig_height * ratio)
            size = (new_width, new_height)
        elif resize_mode['exact']:
            # Exact size
            size = (width, height)
        else:
            # Maintain aspect ratio
            ratio = min(width / orig_width, height / orig_height)
            new_width = int(orig_width * ratio)
            new_height = int(orig_height * ratio)
            size = (new_width, new_height)
        
        # Resize the image
        resized_img = img.resize(size, resize_mode['method'])
        
        # If we're filling but the target size is different, we need to crop
        if resize_mode['fill'] and (size[0] != width or size[1] != height):
            left = (size[0] - width) // 2
            top = (size[1] - height) // 2
            right = left + width
            bottom = top + height
            resized_img = resized_img.crop((left, top, right, bottom))
        
        # Save the resized image
        resized_img.save(dst_path, quality=quality, optimize=True)
        logger.debug(f"Resized: {src_path} -> {dst_path} ({orig_width}x{orig_height} -> {width}x{height})")
        return True
        
    except Exception as e:
        logger.error(f"Error resizing {src_path}: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def process_batch(image_files, src_dir, dst_dir, width, height, resize_mode, quality, overwrite):
    """Process a batch of images."""
    result = {
        'success': 0,
        'failed': 0,
        'skipped': 0
    }
    
    for rel_path in image_files:
        src_path = os.path.join(src_dir, rel_path)
        dst_path = os.path.join(dst_dir, rel_path)
        
        try:
            success = resize_image(src_path, dst_path, width, height, resize_mode, quality, overwrite)
            if success:
                result['success'] += 1
            else:
                result['skipped'] += 1
        except Exception as e:
            logger.error(f"Unexpected error processing {src_path}: {str(e)}")
            result['failed'] += 1
    
    return result

def find_image_files(src_dir, include_extensions, exclude_dirs=None):
    """Find all image files in the source directory."""
    if exclude_dirs is None:
        exclude_dirs = []
    
    # Normalize exclude_dirs to absolute paths
    exclude_dirs = [os.path.abspath(os.path.join(src_dir, d)) for d in exclude_dirs]
    
    image_files = []
    for root, dirs, files in os.walk(src_dir):
        # Skip excluded directories
        if any(os.path.abspath(root).startswith(excluded) for excluded in exclude_dirs):
            continue
        
        for filename in files:
            file_path = os.path.join(root, filename)
            if any(filename.lower().endswith(ext) for ext in include_extensions):
                rel_path = os.path.relpath(file_path, src_dir)
                image_files.append(rel_path)
    
    return image_files

def main():
    """Main entry point for the script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Resize images in a directory.')
    parser.add_argument('src', help='Source directory')
    parser.add_argument('dst', help='Destination directory')
    parser.add_argument('size_str', help='Size string (e.g., "224x224!", "640x360>", "360x360^>")')
    parser.add_argument('--include-exts', type=str, default='.gif,.jpg,.jpeg,.png',
                      help='Comma-separated list of file extensions to include (default: .gif,.jpg,.jpeg,.png)')
    parser.add_argument('--exclude-dirs', type=str, default='',
                      help='Comma-separated list of directories to exclude')
    parser.add_argument('--copy-other-files', action='store_true',
                      help='Copy non-image files to destination')
    parser.add_argument('--num-parallel', type=int, default=os.cpu_count(),
                      help=f'Number of parallel threads (default: {os.cpu_count()})')
    parser.add_argument('--quality', type=int, default=95,
                      help='JPEG quality (1-100, default: 95)')
    parser.add_argument('--overwrite', action='store_true',
                      help='Overwrite existing files')
    parser.add_argument('--log-file', type=str, default=None,
                      help='Path to log file')
    parser.add_argument('--verbose', action='store_true',
                      help='Print detailed progress information')
    
    args = parser.parse_args()
    
    # Configure logging
    configure_logging(args.log_file, args.verbose)
    
    # Process paths
    src_dir = os.path.abspath(args.src)
    dst_dir = os.path.abspath(args.dst)
    
    # Check if source directory exists
    if not os.path.isdir(src_dir):
        logger.error(f"Source directory does not exist: {src_dir}")
        sys.exit(1)
    
    # Parse the size string
    try:
        width, height, resize_mode = parse_size_string(args.size_str)
        logger.info(f"Resizing to {width}x{height} with mode: {resize_mode}")
    except Exception as e:
        logger.error(f"Invalid size string '{args.size_str}': {str(e)}")
        sys.exit(1)
    
    # Process file extensions
    include_extensions = ['.' + ext.lower().strip('.') for ext in args.include_exts.split(',')]
    logger.info(f"Including file extensions: {include_extensions}")
    
    # Process excluded directories
    exclude_dirs = [d.strip() for d in args.exclude_dirs.split(',')] if args.exclude_dirs else []
    if exclude_dirs:
        logger.info(f"Excluding directories: {exclude_dirs}")
    
    # Create destination directory
    os.makedirs(dst_dir, exist_ok=True)
    
    # Find all image files
    logger.info(f"Scanning for images in {src_dir}...")
    start_time = time.time()
    image_files = find_image_files(src_dir, include_extensions, exclude_dirs)
    logger.info(f"Found {len(image_files)} images to process")
    
    # Process images in parallel
    total_results = {'success': 0, 'failed': 0, 'skipped': 0}
    
    # Calculate number of threads to use
    num_workers = min(args.num_parallel, len(image_files))
    if num_workers < 1:
        num_workers = 1
    
    logger.info(f"Processing with {num_workers} worker threads...")
    
    # Create batch sizes
    batch_size = max(1, len(image_files) // (num_workers * 2))
    batches = [image_files[i:i+batch_size] for i in range(0, len(image_files), batch_size)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_batch = {
            executor.submit(
                process_batch, 
                batch, 
                src_dir, 
                dst_dir, 
                width, 
                height, 
                resize_mode,
                args.quality,
                args.overwrite
            ): batch for batch in batches
        }
        
        # Track progress
        completed = 0
        total_batches = len(batches)
        
        for future in concurrent.futures.as_completed(future_to_batch):
            try:
                result = future.result()
                for key in total_results:
                    total_results[key] += result[key]
                
                # Update progress
                completed += 1
                if not args.verbose:
                    progress = completed / total_batches
                    sys.stdout.write(f"\rProgress: [{int(progress*50)*'='}{(50-int(progress*50))*' '}] {progress*100:.1f}%")
                    sys.stdout.flush()
                    
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
    
    # Clear progress bar line
    if not args.verbose:
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()
    
    # Process time
    elapsed_time = time.time() - start_time
    logger.info(f"Image processing completed in {elapsed_time:.2f} seconds")
    logger.info(f"Results: {total_results['success']} successful, {total_results['failed']} failed, {total_results['skipped']} skipped")
    
    # Copy non-image files if requested
    if args.copy_other_files:
        logger.info("Copying non-image files...")
        
        # Find all non-image files
        other_files = []
        for root, _, files in os.walk(src_dir):
            # Skip excluded directories
            if any(os.path.abspath(root).startswith(os.path.abspath(os.path.join(src_dir, d))) for d in exclude_dirs):
                continue
                
            for file in files:
                if not any(file.lower().endswith(ext) for ext in include_extensions):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, src_dir)
                    other_files.append(rel_path)
        
        logger.info(f"Found {len(other_files)} non-image files to copy")
        
        # Copy files
        copied = 0
        for rel_path in other_files:
            src_path = os.path.join(src_dir, rel_path)
            dst_path = os.path.join(dst_dir, rel_path)
            
            # Skip if destination exists and we're not overwriting
            if os.path.exists(dst_path) and not args.overwrite:
                continue
                
            try:
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)
                copied += 1
            except Exception as e:
                logger.error(f"Error copying {src_path}: {str(e)}")
        
        logger.info(f"Copied {copied} non-image files")
    
    # Clean up empty directories
    logger.info("Removing empty directories...")
    empty_dirs_removed = 0
    for root, dirs, files in os.walk(dst_dir, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
                empty_dirs_removed += 1
    logger.info(f"Removed {empty_dirs_removed} empty directories")
    
    # Final summary
    logger.info("Resizing operation complete!")
    logger.info(f"Total images processed: {len(image_files)}")
    logger.info(f"Success: {total_results['success']}")
    logger.info(f"Failed: {total_results['failed']}")
    logger.info(f"Skipped: {total_results['skipped']}")
    logger.info(f"Total time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}")
        logger.critical(traceback.format_exc())
        sys.exit(1)
