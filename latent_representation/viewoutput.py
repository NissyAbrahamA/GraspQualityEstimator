import numpy as np
import open3d as o3d
import os
import burg_toolkit as burg
#
# point_cloud_array = np.load('C:/Users/anizy/Documents/graspqualityestimator_testing/tests/002_master_chef_can_pc.npy')
# point_coordinates = point_cloud_array[:, :3]
# normals = point_cloud_array[:, 3:]
# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(point_coordinates)
# point_cloud.normals = o3d.utility.Vector3dVector(normals)
# o3d.visualization.draw_geometries([point_cloud], point_show_normal=True)
# data = np.load('C:/Users/anizy/Documents/graspqualityestimator_testing/Input_NN/train_latentfeatures/004_sugar_box.npy', allow_pickle=True)
# print(data.shape)
#data = np.load('005_tomato_soup_can_mu0.1.npz')
# keys =data.files
# for key in keys:
#     values = data[key]
#     print(f"Key: {key}, Number of Values: {len(values)}")
# print(data['points'][1000])
# print(data['normals'][1000])
# print(data['angles'][1000])
# print(data['score'][1000])
# #
# with open("objects.txt", "r") as file:
#     object_names = file.read().splitlines()
# print(object_names)
#
# for object_name in object_names:
#     print(object_name)
#     npz_files = [
#         f'{object_name}_random.npz',
#         f'{object_name}_mu0.5.npz',
#         f'{object_name}_mu0.4.npz',
#         f'{object_name}_mu0.3.npz',
#         f'{object_name}_mu0.2.npz',
#         f'{object_name}_mu0.1.npz'
#     ]
#
#     all_data = []
#     for npz_file in npz_files:
#         print(npz_file)
#         data = np.load(npz_file)
#         print(len(data['score']))
#         all_data.append(data)
#
#         merged_data = {}
#         for data in all_data:
#             for key, value in data.items():
#                 if key in merged_data:
#                     merged_data[key] = np.concatenate((merged_data[key], value), axis=0)
#                 else:
#                     merged_data[key] = value
#                 output_dir = 'C:/Users/anizy/OneDrive - Aston University/Documents/fork/burg-toolkit/tests/output'
#
#         output_file = os.path.join(output_dir, f'{object_name}.npz')
#         np.savez(output_file, **merged_data)
# data = np.load('C:/Users/anizy/OneDrive - Aston University/Documents/fork/burg-toolkit/tests/dataset/002_master_chef_can.npz')
# max_score_index = np.argmin(data['score'])
# #
# # # Access corresponding elements using the obtained index
# max_score_point = data['points'][max_score_index]
# max_score_normal = data['normals'][max_score_index]
# max_score_angle = data['angles'][max_score_index]
# max_score_value = data['score'][max_score_index]
# #
# # # Print the values corresponding to the maximum score
# print("Index of max score:", max_score_index)
# print("Max score point:", max_score_point)
# print("Max score normal:", max_score_normal)
# print("Max score angle:", max_score_angle)
# print("Max score value:", max_score_value)
#
#directory = 'C:/Users/anizy/Documents/graspqualityestimator_testing/Input_NN/train_cpscore/debug'

# List all .npz files in the directory
#npz_files = [file for file in os.listdir(directory) if file.endswith('.npz')]

# Iterate through each .npz file
# for npz_file in npz_files:
#     # Load the npz file
#     file_path = os.path.join(directory, npz_file)
#     data = np.load(file_path)
#     data = np.load('//users//2//220269470//dev//graspqualityestimator_testing//Input_NN//debug//002_master_chef_can.npz')
#     keys = data.files
#
#     for key in keys:
#         values = data[key]
#         print(f"File: {npz_file}, Key: {key}, Number of Values: {values}")
#
# import os
# import random
# import numpy as np
# #
# # # Directory containing the .npz files
# directory = 'C:/Users/anizy/Documents/graspqualityestimator_testing/Input_NN/train_cpscore/debug'
#
# # List all .npz files in the directory
# npz_files = [file for file in os.listdir(directory) if file.endswith('.npz')]
#
# # Number of random entries to select
# num_random_entries = 1
#
# # Iterate through each .npz file
# for npz_file in npz_files:
#     # Load the npz file
#     file_path = os.path.join(directory, npz_file)
#     data = np.load(file_path)
#
#     # Get the list of keys in the npz file
#     keys = data.files
#
#     # Create a new dictionary to store the selected random entries
#     new_data = {}
#
#     # Select 1000 random entries for each key
#     for key in keys:
#         values = data[key]
#         num_values = len(values)
#         random_indices = random.sample(range(num_values), num_random_entries)
#         new_data[key] = values[random_indices]
#
#     # Save the new data as an npz file with the same name
#     new_file_path = os.path.join(directory, npz_file)
#     np.savez(new_file_path, **new_data)
#
#     print(f"File: {npz_file} - Selected {num_random_entries} random entries and saved as {new_file_path}")
# #
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import torch

# Load the latent code data on CPU
latent_code = np.load('C:/Users/anizy/Documents/graspqualityestimator_testing/Input_NN/train_latentfeatures/011_banana.npy', allow_pickle=True)

# Apply PCA to reduce dimensionality to 1
pca = PCA(n_components=1)
latent_code_1d = pca.fit_transform(latent_code)

# Create a 64x64x64 grid
grid_size = 64
x, y, z = np.meshgrid(np.arange(grid_size), np.arange(grid_size // 2), np.arange(grid_size))
xyz = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

# Create a scatter plot colored by PCA values
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=latent_code_1d, cmap='viridis')

# Add color bar for reference
cbar = plt.colorbar(sc)
cbar.set_label('PCA Value')

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.title('PCA Visualization of Latent Code')
plt.show()
