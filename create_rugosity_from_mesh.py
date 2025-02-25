import numpy as np
from datetime import datetime
from google.cloud.firestore_v1.field_path import FieldPath
from datetime import datetime
import pandas as pd
import numpy as np
import open3d as o3d
from numpy import genfromtxt

import random


import os

def count_files_with_substring(folder_path, substring):

    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and substring in f]
    return len(files)


def generate_random_rugosity(position, normal, DEFECT, defect_chances):

    random_number = random.randint(1, 100)
    if DEFECT and random_number >= 100 - defect_chances[1]:

        position += defect_chances[2] * normal

    elif random_number >= defect_chances[0]:

        position += defect_chances[2] * normal
        DEFECT = True

    else:

        DEFECT = False

    
    return position, DEFECT

FILENAME = 'Basic_Hollow_Cylinder_main'
FILE_NUMBER = 0
CLASS_DEFECT_CHANCES = [[5, 20, 0.25], [20, 30, 0.25], [30, 50, 0.25], [40, 60, 0.6], [60, 80, 0.8]]
#CLASS_DEFECT_CHANCES = [[5, 20, 0.25], [30, 50, 0.25], [60, 80, 0.8]]
CLASS_NAMES = ['Excelent', 'Good', 'Fair', 'Poor', 'Bad']
BATCH_SIZE = 1000


original_mesh = o3d.io.read_triangle_mesh(f"./3D_meshes/Original/{FILENAME}.stl")

pcd = original_mesh.sample_points_poisson_disk(49152)
print(np.asarray(pcd.points))

pcd = o3d.t.geometry.PointCloud.from_legacy(pcd)



downpcd = pcd.voxel_down_sample(voxel_size=0.03)
downpcd.estimate_normals(max_nn=30, radius=20)


positions = downpcd.point.positions
positions_np = positions.numpy()
normals = downpcd.point.normals
normals_np = normals.numpy()
if count_files_with_substring(f'./Positional_data/Original', f'{FILENAME}_positions.csv') == 0:

    df = pd.DataFrame(positions_np[1:])
    df = pd.concat([df, pd.DataFrame(normals_np[1:])], axis=1)
    df.columns = ['pos_x', 'pos_y', 'pos_z', 'normal_x', 'normal_y', 'normal_z']
    df.to_csv(f'./Positional_data/Original/{FILENAME}_positions.csv', index=False)


FILE_NUMBER = count_files_with_substring(f'./Positional_data/Simulated', FILENAME)

for i in range(0, BATCH_SIZE):

    DEFECT = False
    if i % 5 == 0:

        new_positions_np = positions_np.copy()
        new_normals_np = normals_np.copy()

    for j in range(0, len(new_positions_np)):

        new_positions_np[j], DEFECT = generate_random_rugosity(new_positions_np[j], new_normals_np[j], DEFECT, CLASS_DEFECT_CHANCES[i%5])

    FILE_NUMBER = count_files_with_substring(f'./Positional_data/Simulated', FILENAME)
    df = pd.DataFrame(new_positions_np[1:])
    df = pd.concat([df, pd.DataFrame(new_normals_np[1:])], axis=1)
    df.columns = ['pos_x', 'pos_y', 'pos_z', 'normal_x', 'normal_y', 'normal_z']
    df['predicted_class'] = i%5
    if FILE_NUMBER > 0:

        df.to_csv(f'./Positional_data/Simulated/{FILENAME}_simulated_positions ({FILE_NUMBER}).csv', index=False)

    else:

        df.to_csv(f'./Positional_data/Simulated/{FILENAME}_simulated_positions.csv', index=False)
        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(new_positions_np[1:])
    pcd.normals = o3d.utility.Vector3dVector(new_normals_np[1:])
    print('run Poisson surface reconstruction')
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9)
        
    mesh.compute_vertex_normals()
    if FILE_NUMBER > 0:

        o3d.io.write_triangle_mesh(f"./3D_meshes/Simulated/{FILENAME}_mesh_simulated ({FILE_NUMBER}).stl", mesh)

    else:   

        o3d.io.write_triangle_mesh(f"./3D_meshes/Simulated/{FILENAME}_mesh_simulated.stl", mesh) 


