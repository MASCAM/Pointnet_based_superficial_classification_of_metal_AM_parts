import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh

FILENAME = 'Basic_Hollow_Cylinder_main'
VALIDATION = False
if VALIDATION :
    VALIDATION_BASE_PATH = 'Validation/'
    INPUT_PATH = 'Positional_data/Validation/Simulated/{}_simulated_positions ({}).csv'
    OUTPUT_PATH = 'Features_data/Simulated{}.csv'
else:
    VALIDATION_BASE_PATH = ''
    INPUT_PATH = 'Positional_data/Simulated/{}_simulated_positions ({}).csv'
    OUTPUT_PATH = 'Features_data/Simulated/{}.csv'
    VALIDATION_BASE_PATH = ''

OUTPUT_FILENAME = FILENAME + '_features_extracted'

def load_point_cloud(file_path):
    data = pd.read_csv(file_path).values
    return data

def save_features(features, output_file):
    df = pd.DataFrame(features, columns=[
        'Longest Diagonal', 
        'Thickness', 
        'Fiedler Number', 
        'Maximum Height', 
        'Centroid Distance',
        'Std Height', 
        'Std Centroid Distance', 
        'Std Thickness', 
        'Label'
    ])
    df.to_csv(output_file, index=False)

def longest_diagonal(points):
    z_values = points[:, 2]
    min_z_idx = np.argmin(z_values)
    max_z_idx = np.argmax(z_values)
    min_z_point = points[min_z_idx, :3]
    max_z_point = points[max_z_idx, :3]
    return np.linalg.norm(max_z_point - min_z_point)

def thickness(points, num_points=2048):
    # Sort points by z value in ascending order and select the first num_points
    sorted_points = points[np.argsort(points[:, 2])[:num_points]]
    
    # Assuming the point cloud is cylindrical and symmetrical
    thickness_values = []
    z_range = np.ptp(sorted_points[:, 2]) * 0.1  # 10% of the range of Z values as the small range
    for z in np.linspace(np.min(sorted_points[:, 2]), np.max(sorted_points[:, 2]), num_points):
        z_min = z - z_range / 2
        z_max = z + z_range / 2
        z_points = sorted_points[(sorted_points[:, 2] >= z_min) & (sorted_points[:, 2] <= z_max)]
        if len(z_points) > 1:
            x_min = np.min(z_points[:, 0])
            x_max = np.max(z_points[:, 0])
            thickness_values.append(x_max - x_min)
    return np.mean(thickness_values), np.std(thickness_values)

def fiedler_number(points, num_points=2048):
    # Sort points by z value in descending order and select the top num_points
    sorted_points = points[np.argsort(points[:, 2])[::-1][:num_points]]
    
    # Create a graph from the selected points with distance threshold
    G = nx.Graph()
    
    # Calculate mean distance between points to use as threshold
    distances = []
    for i in range(min(1000, len(sorted_points))):  # Sample 100 points for efficiency
        for j in range(i + 1, min(1000, len(sorted_points))):
            distances.append(np.linalg.norm(sorted_points[i, :3] - sorted_points[j, :3]))
    threshold = np.mean(distances)  # Use mean distance as threshold
    
    # Add edges based on threshold
    for i in range(len(sorted_points)):
        for j in range(i + 1, len(sorted_points)):
            distance = np.linalg.norm(sorted_points[i, :3] - sorted_points[j, :3])
            if distance < threshold:
                G.add_edge(i, j)
    
    # Compute the normalized Laplacian matrix
    L = laplacian(nx.adjacency_matrix(G), normed=True)
    
    try:
        # Compute the two smallest eigenvalues using eigsh
        eigenvalues, _ = eigsh(L, k=2, which='SM', tol=1e-3)
        # Return the second smallest eigenvalue (Fiedler number)
        return eigenvalues[1]
    except:
        # Fallback to dense computation if sparse computation fails
        eigenvalues = np.linalg.eigvalsh(L.toarray())
        eigenvalues.sort()
        return eigenvalues[1]

def maximum_height(points):
    return np.max(points[:, 2])

def centroid_distance(points):
    centroid = np.mean(points[:, :3], axis=0)
    distances = np.linalg.norm(points[:, :3] - centroid, axis=1)
    return np.mean(distances), np.std(distances)

def std_height(points, num_points=2048):
    # Sort points by z value in descending order and select the top num_points
    sorted_points = points[np.argsort(points[:, 2])[::-1][:num_points]]
    heights = sorted_points[:, 2]
    return np.std(heights)

def std_centroid_distance(points, num_points=2048):
    centroid = np.mean(points[:, :3], axis=0)
    distances = np.linalg.norm(points[:, :3] - centroid, axis=1)
    # Sort distances and select the top num_points nearest to the centroid
    sorted_distances = np.sort(distances)[:num_points]
    return np.std(sorted_distances)

def extract_features_from_file(input_file):
    points = load_point_cloud(input_file)
    label = points[0, -1]  # Assuming the label is the same for all points in the file
    std_height_value = std_height(points)
    std_centroid_distance_value = std_centroid_distance(points)
    mean_thickness, std_thickness = thickness(points)
    features = [
        longest_diagonal(points),
        mean_thickness,
        fiedler_number(points),
        maximum_height(points),
        centroid_distance(points)[0],
        std_height_value,
        std_centroid_distance_value,
        std_thickness,
        label
    ]
    return features

def main():
    os.makedirs('Features_data/Simulated', exist_ok=True)
    os.makedirs('Features_data/Validation/Simulated', exist_ok=True)
    all_features = []
    for i in range(1):
        print(f'Processing file {INPUT_PATH.format(FILENAME, i)}')
        if i == 0:
            input_file = f'Positional_data/{VALIDATION_BASE_PATH}Simulated/{FILENAME}_simulated_positions.csv'
        else:
            input_file = INPUT_PATH.format(FILENAME, i)
        features = extract_features_from_file(input_file)
        all_features.append(features)
    output_file = OUTPUT_PATH.format(OUTPUT_FILENAME)
    save_features(all_features, output_file)

if __name__ == "__main__":
    main()