import os
import numpy as np
import h5py
import glob
import re
import pandas as pd
import open3d as o3d

# Constants
NUM_POINT = 2048
INPUT_DIR = f'Segmented_data/Simulated'
OUTPUT_DIR = f'Pre_Processed_data/3_classes/Simulated/HDF5/'

def convert_5class_to_3class(labels):
    """Convert 5-class labels to 3-class labels by removing classes 1 and 3."""
    # Create a mapping array
    mapping = np.array([0, -1, 1, -1, 2], dtype=np.int8)
    
    # Apply mapping
    new_labels = mapping[labels]
    
    # Verify no -1 values remain
    if np.any(new_labels == -1):
        print("Warning: Found labels for classes that should be ignored")
    
    return new_labels

def save_h5_data_label_normal(h5_filename, data, label, normal, 
        data_dtype='float32', label_dtype='uint8', normal_dtype='float32'):
    h5_fout = h5py.File(h5_filename, 'w')
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'normal', data=normal,
            compression='gzip', compression_opts=4,
            dtype=normal_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()

def voxel_downsample(points, voxel_size):
    """
    Reduz a nuvem de pontos usando voxelização
    
    Args:
        points: numpy array de shape (N, 3) com os pontos 3D
        voxel_size: tamanho do voxel para amostragem
    Returns:
        numpy array com pontos reduzidos
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(downsampled.points)

def process_csv_to_h5(csv_file, h5_file, num_points):
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Extract points, normals, and label (single label for all points)
    points = df[['pos_x', 'pos_y', 'pos_z']].values
    normals = df[['normal_x', 'normal_y', 'normal_z']].values
    label = df['predicted_class'].iloc[0]  # Get the single label value
    
    # Skip files with labels 1 or 3
    if label in [1, 3]:
        print(f"Skipping file {csv_file} with label {label}")
        return False
        
    # Convert labels 2 and 4 to 1 and 2 respectively
    if label == 2:
        label = 1
    elif label == 4:
        label = 2
    # label 0 remains 0
    
    # Create label array with shape (2048, 1)
    labels = np.full((num_points, 1), label, dtype=np.uint8)
    
    # Process point cloud to have exactly num_points points
    if len(points) > num_points:
        # Use voxel downsampling
        voxel_size = 0.02  # Start with this size
        while len(points) > num_points:
            points = voxel_downsample(points, voxel_size)
            voxel_size *= 1.1  # Increase voxel size if needed
            
        # If we still have too many points, randomly sample
        if len(points) > num_points:
            idx = np.random.choice(len(points), num_points, replace=False)
            points = points[idx]
            normals = normals[idx]
    else:
        # If we have too few points, randomly duplicate some points
        idx = np.random.choice(len(points), num_points, replace=True)
        points = points[idx]
        normals = normals[idx]
    
    # Normalize point cloud to unit sphere
    centroid = np.mean(points, axis=0)
    points = points - centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2, axis=1)))
    points = points / furthest_distance
    
    # Normalize normals to unit length
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    
    # Save to H5 file
    save_h5_data_label_normal(h5_file, points, labels, normals)
    return True

def unite_h5_files(h5_files, output_file):
    data_list = []
    normals_list = []
    labels_list = []
    
    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as f_in:
            # Check for NaN values in all datasets
            data = f_in['data'][:]
            normal = f_in['normal'][:]
            label = f_in['label'][:]  # This should have shape (2048, 1)
            
            if np.any(np.isnan(data)) or np.any(np.isnan(normal)):
                print(f"Warning: NaN values found in {h5_file}")
                continue
                
            data_list.append(data)
            normals_list.append(normal)
            labels_list.append(label)
    
    # Stack the arrays while maintaining the label shape
    combined_data = np.stack(data_list, axis=0)  # Shape: (n_files, 2048, 3)
    combined_normals = np.stack(normals_list, axis=0)  # Shape: (n_files, 2048, 3)
    combined_labels = np.stack(labels_list, axis=0)  # Shape: (n_files, 2048, 1)
    
    with h5py.File(output_file, 'w') as f_out:
        f_out.create_dataset('data', data=combined_data)
        f_out.create_dataset('normal', data=combined_normals)
        f_out.create_dataset('label', data=combined_labels)

def create_unified_h5_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all CSV files in the input directory
    csv_files = glob.glob(os.path.join(input_dir, '**/*.csv'), recursive=True)
    print(f"Found {len(csv_files)} CSV files")
    
    # Regular expression to match pattern names
    pattern_regex = r'((?:Basic_)?(?:Hollow|Holow)_[A-Za-z_]+)_(?:main_segmented|segmented)_positions'
    
    # Dictionary to store files for each pattern
    pattern_files = {}
    
    # Process each file
    for csv_file in csv_files:
        # Extract pattern name from filename
        match = re.search(pattern_regex, csv_file)
        if not match:
            continue
            
        pattern_name = match.group(1)
        print(f"Processing {pattern_name} file: {csv_file}")
        
        # Initialize pattern files list if not exists
        if pattern_name not in pattern_files:
            pattern_files[pattern_name] = []
        
        # Create individual H5 file
        h5_file = os.path.join(output_dir, f'{os.path.basename(csv_file)[:-4]}.h5')
        if process_csv_to_h5(csv_file, h5_file, NUM_POINT):
            pattern_files[pattern_name].append(h5_file)
        else:
            print(f"Skipping {csv_file} due to invalid label")
    
    # Create train_files.txt and test_files.txt
    train_files = []
    test_files = []
    
    # Create unified files for each pattern
    for pattern_name, files in pattern_files.items():
        # Split files into train and test
        split_idx = int(len(files) * 0.8)
        pattern_train_files = files[:split_idx]
        pattern_test_files = files[split_idx:]
        
        # Create unified train file
        train_output = os.path.join(output_dir, f'{pattern_name.lower()}_train.h5')
        unite_h5_files(pattern_train_files, train_output)
        train_files.append(train_output)
        
        # Create unified test file
        test_output = os.path.join(output_dir, f'{pattern_name.lower()}_test.h5')
        unite_h5_files(pattern_test_files, test_output)
        test_files.append(test_output)
        
        print(f"Created unified files for {pattern_name}")
        print(f"Train: {train_output}")
        print(f"Test: {test_output}")
    
    # Write file lists
    with open(os.path.join(output_dir, 'train_files.txt'), 'w') as f:
        for file in train_files:
            f.write(file + '\n')
    
    with open(os.path.join(output_dir, 'test_files.txt'), 'w') as f:
        for file in test_files:
            f.write(file + '\n')

if __name__ == '__main__':
    create_unified_h5_files(INPUT_DIR, OUTPUT_DIR) 
