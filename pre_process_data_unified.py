import os
import numpy as np
import h5py
import glob
import re
import pandas as pd
import open3d as o3d

# Constants
NUM_POINT = 2048
INPUT_DIR = f'Segmented_data/Original/'
OUTPUT_DIR = f'Pre_Processed_data/Original/HDF5/'


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
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(downsampled.points)

def process_point_cloud(pcd, n_points=2048, method='voxel', labels=None):
    """Process point cloud to have exactly n_points"""
    points = np.asarray(pcd)
    
    if method == 'voxel':
        # Start with a relatively large voxel size and adjust if needed
        voxel_size = 0.05
        max_attempts = 10
        attempt = 0
        
        while attempt < max_attempts:
            reduced_points = voxel_downsample(points, voxel_size)
            if len(reduced_points) > n_points:
                # If we have too many points, increase voxel size
                voxel_size *= 1.2
            elif len(reduced_points) < n_points:
                # If we have too few points, decrease voxel size
                voxel_size *= 0.8
            else:
                break
            attempt += 1
        
        # After attempts, if we still don't have exact number, randomly sample
        if len(reduced_points) != n_points:
            if len(reduced_points) > n_points:
                # Randomly select n_points
                indices = np.random.choice(len(reduced_points), n_points, replace=False)
                reduced_points = reduced_points[indices]
            else:
                # Randomly duplicate points until we have n_points
                additional_points = n_points - len(reduced_points)
                indices = np.random.choice(len(reduced_points), additional_points)
                reduced_points = np.vstack((reduced_points, reduced_points[indices]))
        
        # Estimate normals
        pcd_reduced = o3d.geometry.PointCloud()
        pcd_reduced.points = o3d.utility.Vector3dVector(reduced_points)
        pcd_reduced.estimate_normals()
        reduced_normals = np.asarray(pcd_reduced.normals)
        
        # Ensure we have exactly n_points
        assert len(reduced_points) == n_points, f"Expected {n_points} points, got {len(reduced_points)}"
        assert len(reduced_normals) == n_points, f"Expected {n_points} normals, got {len(reduced_normals)}"
    
    reduced_labels = np.full([n_points, 1], labels[0])
    return reduced_points, reduced_normals, reduced_labels

def ensure_label_shape(reduced_labels, target_shape=(2048, 3)):
    current_shape = reduced_labels.shape
    if current_shape[0] < target_shape[0]:
        num_entries_to_add = target_shape[0] - current_shape[0]
        additional_entries = np.full((num_entries_to_add, target_shape[1]), reduced_labels[0])
        reduced_labels = np.vstack((reduced_labels, additional_entries))
    return reduced_labels

def process_csv_to_h5(csv_file, h5_file, num_point):
    """Process a single CSV file and save as H5"""
    try:
        # Load the CSV file
        data = pd.read_csv(csv_file)

        # Extract point cloud data and normals
        points = data[['pos_x', 'pos_y', 'pos_z']].values
        normals = data[['normal_x', 'normal_y', 'normal_z']].values
        labels = data['predicted_class'].values

        if points.shape[0] > num_point:
            points, normals, labels = process_point_cloud(points, num_point, 'voxel', labels)
        elif points.shape[0] < num_point:
            points = ensure_label_shape(points, (num_point, 3))
            normals = ensure_label_shape(normals, (num_point, 3))
        
        labels = np.full([num_point, 1], labels[0])
        
        # Verify shapes before saving
        assert points.shape == (num_point, 3), f"Points shape {points.shape} != ({num_point}, 3)"
        assert normals.shape == (num_point, 3), f"Normals shape {normals.shape} != ({num_point}, 3)"
        assert labels.shape == (num_point, 1), f"Labels shape {labels.shape} != ({num_point}, 1)"
        
        save_h5_data_label_normal(h5_file, points, labels, normals)
        print(f"Successfully processed {csv_file} -> {h5_file}")
        
    except Exception as e:
        print(f"Error processing {csv_file}: {str(e)}")
        raise

def unite_h5_files(h5_files, output_file):
    data_list = []
    normals_list = []
    labels_list = []
    
    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as f_in:
            for key in f_in.keys():
                data = f_in[key][:]
                            # Check for NaN values

                if np.any(np.isnan(data)):

                    print(f"Warning: NaN values found in {h5_file}")

                    continue
                if key == 'data':
                    data_list.append(data[:, :3])
                elif key == 'normal':
                    normals_list.append(data[:, :3])
                elif key == 'label':
                    labels_list.append(data)
    
    with h5py.File(output_file, 'w') as f_out:
        f_out.create_dataset('data', data=np.array(data_list))
        f_out.create_dataset('normal', data=np.array(normals_list))
        f_out.create_dataset('label', data=np.array(labels_list))

def create_unified_h5_files(input_dir, output_dir):
    """
    Create unified H5 files from all CSV files in the input directory.
    Each pattern type will be combined into a single H5 file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store files for each pattern
    pattern_files = {}
    
    # Get all CSV files in the input directory
    csv_files = glob.glob(os.path.join(input_dir, '**/*.csv'), recursive=True)
    print(f"Found {len(csv_files)} CSV files")
    
    # Regular expression to match pattern names
    pattern_regex = r'((?:Basic_)?(?:Hollow|Holow)_[A-Za-z_]+)_(?:main_segmented|segmented)_positions'
    
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
        process_csv_to_h5(csv_file, h5_file, NUM_POINT)
        pattern_files[pattern_name].append(h5_file)
    
    # Lists for train and test files
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
    
    # Save train and test file lists
    with open(os.path.join(output_dir, 'train_files.txt'), 'w') as f:
        f.write('\n'.join(train_files))
    
    with open(os.path.join(output_dir, 'test_files.txt'), 'w') as f:
        f.write('\n'.join(test_files))
    
    print(f"\nProcessing complete:")
    print(f"Created unified H5 files for {len(pattern_files)} patterns")
    print(f"Train files: {len(train_files)}")
    print(f"Test files: {len(test_files)}")

def main():
    # Create unified H5 files
    create_unified_h5_files(INPUT_DIR, OUTPUT_DIR)

if __name__ == '__main__':
    main() 