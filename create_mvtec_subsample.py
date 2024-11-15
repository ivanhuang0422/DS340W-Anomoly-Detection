import os
import shutil
import random
import argparse
from pathlib import Path
import glob

def create_subsample(source_dir, target_dir, sample_ratio=0.2, random_seed=42):
    """
    Create a subsampled version of the MVTec AD dataset.
    
    Args:
        source_dir (str): Path to the original MVTec AD dataset
        target_dir (str): Path where the subsampled dataset will be created
        sample_ratio (float): Ratio of images to keep (0.0 to 1.0)
        random_seed (int): Random seed for reproducibility
    """
    random.seed(random_seed)
    
    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    # Get all object categories
    object_categories = [d for d in os.listdir(source_dir) 
                        if os.path.isdir(os.path.join(source_dir, d))]
    
    for category in object_categories:
        print(f"Processing category: {category}")
        
        # Create category directory
        category_target_dir = os.path.join(target_dir, category)
        os.makedirs(category_target_dir, exist_ok=True)
        
        # Process train directory
        train_source = os.path.join(source_dir, category, "train")
        train_target = os.path.join(category_target_dir, "train")
        if os.path.exists(train_source):
            os.makedirs(train_target, exist_ok=True)
            
            # Process each class in train (usually just 'good')
            for class_name in os.listdir(train_source):
                class_source = os.path.join(train_source, class_name)
                class_target = os.path.join(train_target, class_name)
                os.makedirs(class_target, exist_ok=True)
                
                # Get all images and subsample
                images = glob.glob(os.path.join(class_source, "*.*"))
                sample_size = max(1, int(len(images) * sample_ratio))
                selected_images = random.sample(images, sample_size)
                
                # Copy selected images
                for img_path in selected_images:
                    shutil.copy2(img_path, class_target)
        
        # Process test directory
        test_source = os.path.join(source_dir, category, "test")
        test_target = os.path.join(category_target_dir, "test")
        if os.path.exists(test_source):
            os.makedirs(test_target, exist_ok=True)
            
            # Process each defect type in test
            for defect_type in os.listdir(test_source):
                defect_source = os.path.join(test_source, defect_type)
                defect_target = os.path.join(test_target, defect_type)
                os.makedirs(defect_target, exist_ok=True)
                
                # Get all images and subsample
                images = glob.glob(os.path.join(defect_source, "*.*"))
                sample_size = max(1, int(len(images) * sample_ratio))
                selected_images = random.sample(images, sample_size)
                
                # Copy selected images
                for img_path in selected_images:
                    shutil.copy2(img_path, defect_target)
        
        # Process ground truth directory if it exists
        ground_truth_source = os.path.join(source_dir, category, "ground_truth")
        ground_truth_target = os.path.join(category_target_dir, "ground_truth")
        if os.path.exists(ground_truth_source):
            os.makedirs(ground_truth_target, exist_ok=True)
            
            # Process each defect type in ground truth
            for defect_type in os.listdir(ground_truth_source):
                defect_source = os.path.join(ground_truth_source, defect_type)
                defect_target = os.path.join(ground_truth_target, defect_type)
                os.makedirs(defect_target, exist_ok=True)
                
                # Get all mask images and copy corresponding ones
                for img_path in glob.glob(os.path.join(defect_source, "*.*")):
                    # Only copy ground truth for selected test images
                    img_name = os.path.basename(img_path)
                    corresponding_test = os.path.join(test_target, defect_type, 
                                                    img_name.replace("_mask", ""))
                    if os.path.exists(corresponding_test):
                        shutil.copy2(img_path, defect_target)

def main():
    parser = argparse.ArgumentParser(description='Create a subsampled version of MVTec AD dataset')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to the original MVTec AD dataset')
    parser.add_argument('--target', type=str, required=True,
                        help='Path where the subsampled dataset will be created')
    parser.add_argument('--ratio', type=float, default=0.2,
                        help='Ratio of images to keep (0.0 to 1.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    print(f"Creating subsample with ratio {args.ratio}")
    print(f"Source directory: {args.source}")
    print(f"Target directory: {args.target}")
    
    create_subsample(args.source, args.target, args.ratio, args.seed)
    print("Subsampling completed successfully!")

if __name__ == '__main__':
    main()