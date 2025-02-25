import os
import numpy as np
import pandas as pd
import open3d as o3d

FILENAME = 'Basic_Hollow_Cylinder_main'
VALIDATION = False
if VALIDATION :
    VALIDATION_BASE_PATH = 'Validation/'
    INPUT_PATH = 'Positional_data/Validation/Simulated/{}_simulated_positions ({}).csv'
    OUTPUT_PATH = 'Segmented_data/Validation/Original/{} ({}).csv'
else:
    INPUT_PATH = 'Positional_data/Simulated/{}_simulated_positions ({}).csv'
    OUTPUT_PATH = 'Segmented_data/Simulated/{} ({}).csv'
    VALIDATION_BASE_PATH = ''

OUTPUT_FILENAME = FILENAME + '_segmented_positions'

def load_point_cloud(file_path):
    data = pd.read_csv(file_path).values
    return data

def save_point_cloud(data, file_path):
    df = pd.DataFrame(data, columns=['pos_x', 'pos_y', 'pos_z', 'normal_x', 'normal_y', 'normal_z', 'predicted_class'])
    #print(df)
    df.to_csv(file_path, header=True, index=False)

def normalize_bottom_points(segment, k_neighbors=5, density_radius=0.005):
    """
    Normalize bottom points based on point density and adjust outliers.
    Preserves all columns including labels/predicted_class.
    """
    if len(segment) == 0:
        return np.zeros((2048, segment.shape[1]))
    
    # Create point cloud and KDTree
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(segment[:, :3])
    pcd.normals = o3d.utility.Vector3dVector(segment[:, 3:6])
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # Find points density and Z distribution
    z_values = segment[:, 2]
    min_z = np.min(z_values)
    max_z = np.max(z_values)
    z_range = max_z - min_z
    
    # Calculate density for bottom region
    bottom_region_height = z_range * 0.1
    bottom_points_mask = z_values <= (min_z + bottom_region_height)
    bottom_points = segment[bottom_points_mask]
    
    if len(bottom_points) > 0:
        # Process bottom points
        bottom_pcd = o3d.geometry.PointCloud()
        bottom_pcd.points = o3d.utility.Vector3dVector(bottom_points[:, :3])
        bottom_pcd.normals = o3d.utility.Vector3dVector(bottom_points[:, 3:6])
        
        # Voxel downsample bottom points for uniform density
        voxel_size = density_radius
        bottom_pcd_down = bottom_pcd.voxel_down_sample(voxel_size=voxel_size)
        
        # Get target Z from downsampled bottom points
        bottom_points_down = np.asarray(bottom_pcd_down.points)
        target_z = np.median(bottom_points_down[:, 2])
        
        # Adjust sparse bottom points to target Z
        for i in range(len(segment)):
            if z_values[i] <= (min_z + bottom_region_height):
                [k, _, _] = kdtree.search_radius_vector_3d(segment[i, :3], density_radius)
                if k < k_neighbors:
                    segment[i, 2] = target_z
    
    # Create point cloud from adjusted segment
    adjusted_pcd = o3d.geometry.PointCloud()
    adjusted_pcd.points = o3d.utility.Vector3dVector(segment[:, :3])
    adjusted_pcd.normals = o3d.utility.Vector3dVector(segment[:, 3:6])
    
    # Voxel downsample the entire segment
    voxel_size = density_radius * 2
    downsampled_pcd = adjusted_pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # Convert back to numpy arrays
    points = np.asarray(downsampled_pcd.points)
    normals = np.asarray(downsampled_pcd.normals)
    
    # Get labels from original points (nearest neighbor)
    if segment.shape[1] > 6:  # If we have labels
        labels = segment[:, 6:]  # Get all columns after position and normals
        
        # Find nearest neighbors for label assignment
        tree = o3d.geometry.KDTreeFlann(adjusted_pcd)
        new_labels = []
        for point in points:
            [_, idx, _] = tree.search_knn_vector_3d(point, 1)
            new_labels.append(labels[idx[0]])
        new_labels = np.array(new_labels)
        
        # Combine points, normals and labels
        combined = np.hstack([points, normals, new_labels])
    else:
        combined = np.hstack([points, normals])
    
    # Ensure exactly 2048 points
    if len(combined) > 2048:
        indices = np.random.choice(len(combined), 2048, replace=False)
        combined = combined[indices]
    elif len(combined) < 2048:
        extra_needed = 2048 - len(combined)
        extra_indices = np.random.choice(len(combined), extra_needed, replace=True)
        extra_points = combined[extra_indices]
        combined = np.vstack([combined, extra_points])
    
    return combined

def segment_point_cloud(data, num_segments=24, points_per_segment=2048):
    z_values = data[:, 2]
    min_z, max_z = np.min(z_values), np.max(z_values)
    segment_height = (max_z - min_z) / num_segments

    segmented_data = []
    for i in range(num_segments):
        z_min = min_z + i * segment_height
        z_max = z_min + segment_height
        segment = data[(z_values >= z_min) & (z_values < z_max)].copy()
        
        if segment.shape[0] > 0:
            # Normalize bottom points based on density
            segment = normalize_bottom_points(segment)
            
            # Normalize Z positions within segment
            segment_z_min = np.min(segment[:, 2])
            segment[:, 2] = segment[:, 2] - segment_z_min
        
        segmented_data.append(segment)
    
    return segmented_data

def apply_icp(source, target):
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source[:, :3])
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target[:, :3])

    threshold = 0.02
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    transformation = reg_p2p.transformation
    source[:, :3] = np.asarray(source_pcd.transform(transformation).points)
    return source

def process_file(input_file, output_base, num_segments=24, points_per_segment=2048, file_number=0):
    data = load_point_cloud(input_file)
    segmented_data = segment_point_cloud(data, num_segments, points_per_segment)

    # Apply ICP alignment based on the first segmented part
    reference_segment = segmented_data[0]
    for i in range(1, num_segments):
        segmented_data[i] = apply_icp(segmented_data[i], reference_segment)

    # Save segmented parts to separate CSV files
    for i, segment in enumerate(segmented_data):
        if (file_number == 0 and i == 0):
            output_file = f'Segmented_data/{VALIDATION_BASE_PATH}Original/{FILENAME}_segmented_positions.csv'
        else:
            output_file = output_base.format(OUTPUT_FILENAME, file_number * num_segments + i)
        #print(i)
        save_point_cloud(segment, output_file)

def main():
    os.makedirs('Segmented_data/Simulated', exist_ok=True)
    os.makedirs('Segmented_data/Validation/Simulated', exist_ok=True)
    for i in range(1000):
        #print(i)
        
        if i == 0:

            input_file = f'Positional_data/{VALIDATION_BASE_PATH}Simulated/{FILENAME}_simulated_positions.csv'

        else:

            input_file = INPUT_PATH.format(FILENAME, i)

        process_file(input_file, OUTPUT_PATH, 24, 2048, i)

if __name__ == "__main__":
    main()