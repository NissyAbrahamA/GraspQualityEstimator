import os
from timeit import default_timer as timer
import numpy as np
import configparser
import yaml
import open3d as o3d

import burg_toolkit as burg

SAVE_FILE = os.path.join('..', 'sampled_grasps.npy')


def test_distance_and_coverage():
    # testing the distance function
    initial_translations = np.random.random((50, 3))
    gs = burg.grasp.GraspSet.from_translations(initial_translations)

    theta = 0 / 180 * np.pi
    rot_mat = np.asarray([[1, 0, 0],
                          [0, np.cos(theta), -np.sin(theta)],
                          [0, np.sin(theta), np.cos(theta)]])

    grasp = gs[0]
    grasp.translation = np.asarray([0, 0, 0.003])
    grasp.rotation_matrix = rot_mat
    gs[0] = grasp
    print(grasp)

    theta = 15 / 180 * np.pi
    rot_mat = np.asarray([[np.cos(theta), 0, np.sin(theta)],
                          [0, 1, 0],
                          [-np.sin(theta), 0, np.cos(theta)]])

    grasp = gs[1]
    grasp.translation = np.asarray([0, 0, 0])
    grasp.rotation_matrix = rot_mat
    gs[1] = grasp
    print(grasp)

    print('average gripper point distances between 20 and 50 elem graspset')
    print(burg.metrics.avg_gripper_point_distances(gs[0:20], gs).shape)

    dist = burg.metrics.combined_distances(gs[0], gs[1])
    print('computation of pairwise_distances (15 degree and 3 mm)', dist.shape, dist)

    t1 = timer()
    print('computation of coverage 20/50:', burg.metrics.coverage_brute_force(gs, gs[0:20]))
    print('this took:', timer() - t1, 'seconds')

    t1 = timer()
    print('coverage kd-tree:', burg.metrics.coverage(gs, gs[0:20], print_timings=True))
    print('this took:', timer() - t1, 'seconds')

    grasp_folder = 'e:/datasets/21_ycb_object_grasps/'
    grasp_file = '061_foam_brick/grasps.h5'
    grasp_set, com = burg.io.read_grasp_file_eppner2019(os.path.join(grasp_folder, grasp_file))

    t1 = timer()
    # this is unable to allocate enough memory for len(gs)=500
    #print('computation of coverage 20/50:', gdt.grasp.coverage_brute_force(grasp_set, gs))
    #print('this took:', timer() - t1, 'seconds')

    t1 = timer()
    print('coverage kd-tree:', burg.metrics.coverage(grasp_set, gs, print_timings=True))
    print('in total, this took:', timer() - t1, 'seconds')



def test_new_antipodal_grasp_sampling():
    redo = []
    gripper_model = burg.gripper.ParallelJawGripper(finger_length=0.03,
                                                    finger_thickness=0.003)
    with open("objects.txt", "r") as file:
        object_names = file.read().splitlines()

    for object_name in object_names:
        mesh_fn = f"C:/Users/anizy/OneDrive - Aston University/Desktop/Dissertation/pointcloud/models/ycb/{object_name}/google_16k/textured.obj"

        #mesh_fn= '../input/002_master_chef_can_pointcloud.ply'
        #obj_name = os.path.splitext(os.path.basename(mesh_fn))[0]
        #mesh = o3d.io.read_triangle_mesh(mesh_fn)
        #n_sample = np.max([500, 1000, len(mesh.triangles)])
        #pc_with_normals = burg.util.o3d_pc_to_numpy(burg.mesh_processing.poisson_disk_sampling(mesh, n_points=n_sample))
        #pc_with_normals = burg.util.o3d_pc_to_numpy(burg.mesh_processing.poisson_disk_sampling(mesh))
        #np.save(object_name + '_pc.npy', pc_with_normals)

        ags = burg.sampling.AntipodalGraspSampler()
        ags.mesh = burg.io.load_mesh(mesh_fn)
        ags.gripper = gripper_model
        ags.n_orientations = 1
        ags.verbose = True
        ags.max_targets_per_ref_point = 1
        graspset, contacts, pc_with_normals = ags.sample(500)
        np.save(object_name + '_pc.npy', pc_with_normals)
        #print(contacts[0])
        #contacts_with_mscore = np.copy(contacts)
        # for c in contacts:
        #     nc = np.concatenate((c[:2], np.expand_dims(c[2], axis=0)))
        #     contacts_with_mscore.append(nc)
        # contacts_with_mscore = np.array(contacts_with_mscore)
        #contacts_with_mscore[:, 2:5] = contacts_with_mscore[:, 2:3]
        # contacts_with_mscore = np.array([list(np.unique(row)) for row in contacts])
        # print(contacts_with_mscore[0])
        graspset.scores = ags.check_collisions(graspset, use_width=False)  # need to install python-fcl
        filtered_grasps_contacts = {}
        filtered_grasps_contacts['points'] = []
        filtered_grasps_contacts['normals'] = []
        filtered_grasps_contacts['angles'] = []
        filtered_grasps_contacts['score'] = []
        for grasp, score, contact,normals,angles,cp_score in zip(graspset, graspset.scores, contacts['points'],contacts['normals'],contacts['angles'],contacts['score']):
            if score == 1.0:
                filtered_grasps_contacts['points'].append(contact)
                filtered_grasps_contacts['normals'].append(normals)
                filtered_grasps_contacts['angles'].append(angles)
                filtered_grasps_contacts['score'].append(cp_score)

        if len(filtered_grasps_contacts['points']) < 90:
            redo.append(object_name)
        print(redo)
        np.savez(object_name + "_test.npz", **filtered_grasps_contacts)

def test_antipodal_grasp_random_sampling():
    redo = []
    gripper_model = burg.gripper.ParallelJawGripper(finger_length=0.03,
                                                    finger_thickness=0.003)
    with open("objects.txt", "r") as file:
        object_names = file.read().splitlines()

    for object_name in object_names:
        mesh_fn = f"C:/Users/anizy/OneDrive - Aston University/Desktop/Dissertation/pointcloud/models/ycb/{object_name}/google_16k/textured.obj"
        ags = burg.sampling.AntipodalGraspSampler()
        ags.mesh = burg.io.load_mesh(mesh_fn)
        ags.gripper = gripper_model
        ags.n_orientations = 1
        ags.verbose = True
        ags.max_targets_per_ref_point = 1
        graspset, contacts = ags.randomsample(5000)
        graspset.scores = ags.check_collisions(graspset, use_width=False)
        filtered_grasps_contacts = {}
        filtered_grasps_contacts['points'] = []
        filtered_grasps_contacts['normals'] = []
        filtered_grasps_contacts['angles'] = []
        filtered_grasps_contacts['score'] = []
        for grasp, score, contact,normals,angles,cp_score in zip(graspset, graspset.scores, contacts['points'],contacts['normals'],contacts['angles'],contacts['score']):
            if score == 1.0:
                filtered_grasps_contacts['points'].append(contact)
                filtered_grasps_contacts['normals'].append(normals)
                filtered_grasps_contacts['angles'].append(angles)
                filtered_grasps_contacts['score'].append(cp_score)

        if len(filtered_grasps_contacts['points']) < 1500:
            redo.append(object_name)
        print(redo)
        np.savez(object_name + "_random.npz", **filtered_grasps_contacts)



def test_rotation_to_align_vectors():
    vec_a = np.array([1, 0, 0])
    vec_b = np.array([0, 1, 0])
    r = burg.util.rotation_to_align_vectors(vec_a, vec_b)
    print('vec_a', vec_a)
    print('vec_b', vec_b)
    print('R*vec_a', np.dot(r, vec_a.reshape(3, 1)))

    vec_a = np.array([1, 0, 0])
    vec_b = np.array([-1, 0, 0])
    r = burg.util.rotation_to_align_vectors(vec_a, vec_b)
    print('vec_a', vec_a)
    print('vec_b', vec_b)
    print('R*vec_a', np.dot(r, vec_a.reshape(3, 1)))


def show_grasp_pose_definition():
    gs = burg.grasp.GraspSet.from_translations(np.asarray([0, 0, 0]).reshape(-1, 3))
    gripper = burg.gripper.ParallelJawGripper(finger_length=0.03, finger_thickness=0.003, opening_width=0.05)
    burg.visualization.show_grasp_set([o3d.geometry.TriangleMesh.create_coordinate_frame(0.02)],
                                      gs, gripper=gripper)


def test_angles():
    vec_a = np.array([-0.037533,  0.00233,  -0.041748])
    vec_b = np.array([0.6694,  -0.04297, 0.7417 ])
    mask = np.array([0])
    print(mask)
    a = burg.util.angle(vec_a, vec_b, sign_array=mask, as_degree=False)
    print(a)
    print(mask)


def test_cone_sampling():
    axis = [0, 1, 0]
    angle = np.pi/4
    rays = burg.sampling.rays_within_cone(axis, angle, n=100)

    print(rays.shape)


def visualise_perturbations():
    positions = np.zeros((3, 3))
    positions[0] = [0.3, 0, 0]
    positions[1] = [0, 0, 0.3]

    gs = burg.grasp.GraspSet.from_translations(positions)
    gs_perturbed = burg.sampling.grasp_perturbations(gs, radii=[5, 10, 15])
    gripper = burg.gripper.ParallelJawGripper(finger_length=0.03, finger_thickness=0.003, opening_width=0.05)
    burg.visualization.show_grasp_set([], gs_perturbed, gripper=gripper)

def test_scoring_for_grippers():
    #how-to convert this mesh to the scene data  in the visualisation
    #point cloud from scenes
    data = np.load("C:/Users/anizy/OneDrive - Aston University/Documents/GraspQualityEstimator/scenes/002_master_chef_can/full_point_cloud.npz")
    # data = np.load(
    #     "C:/Users/anizy/OneDrive - Aston University/Documents/fork/burg-toolkit/tests/dataset/002_master_chef_can.npz")
    points = data['points']
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    #mesh_fn = "C:/Users/anizy/OneDrive - Aston University/Documents/GraspQualityEstimator/scenes/002_master_chef_can/pointcloud.ply"
    #mesh_fn = "C:/Users/anizy/OneDrive - Aston University/Desktop/Dissertation/pointcloud/models/ycb/002_master_chef_can/google_16k/textured.obj"
    #mesh = burg.io.load_mesh(mesh_fn)
    # cp1 = np.array([-0.06323,   -0.02559,    0.000794])
    # cp2 = np.array([0.0005007, -0.0356,     0.003418])
    # normal1 = np.array([0.8794,  0.3086, -0.3623])
    # normal2 = np.array([-0.1969,  0.2625, -0.945])
    cp1 = np.array([0.12141492, 0.11315102, 0.21749111])
    cp2 = np.array([0.13996961, 0.19805092, 0.10788723])
    normal1 = np.array([-0.12878418, -0.21923828,  0.96728516])
    normal2 = np.array([-0.20227051,  0.97949219, -0.00253677])
    d = (cp2-cp1)

    angle_cp1, angle_cp2, score = burg.util.calc_score(d,normal1, normal2)
    print(angle_cp1)
    print(angle_cp2)
    print('score:' + str(score))
    burg.visualization.plot_contacts_normals(point_cloud,cp1,cp2,normal1,normal2, score)


def transform_point(point, transformation_matrix):
    homogeneous_point = np.append(point, 1)
    #append 0 for normals
    transformed_homogeneous_point = np.dot(transformation_matrix, homogeneous_point)
    transformed_point = transformed_homogeneous_point[:3] / transformed_homogeneous_point[3]
    #vectors verify this above step
    return transformed_point

def transform_normal(normal, transformation_matrix):
    homogeneous_normal = np.append(normal, 0)  # Append 0 for the normal
    transformed_homogeneous_normal = np.dot(transformation_matrix, homogeneous_normal)
    transformed_normal = transformed_homogeneous_normal[:3] / 1
    return transformed_normal

def augmented_tranformation():
    folder_path = "C:/Users/anizy/OneDrive - Aston University/Documents/GraspQualityEstimator/tests/dataset/"
    yaml_fol = "C:/Users/anizy/OneDrive - Aston University/Documents/GraspQualityEstimator/scenes/"
    output_folder = "C:/Users/anizy/OneDrive - Aston University/Documents/GraspQualityEstimator/transformed_data/"

    if os.path.exists(folder_path):
        file_list = os.listdir(folder_path)
        for file_name in file_list:
            yaml_fol_path = os.path.join(yaml_fol, file_name.replace('.npz', '/'))
            if os.path.isdir(yaml_fol_path):
                yaml_file_path = os.path.join(yaml_fol_path, "scene.yaml")
                with open(yaml_file_path, "r") as yaml_file:
                    scene_data = yaml.load(yaml_file, Loader=yaml.FullLoader)
                    if 'objects' in scene_data and len(scene_data['objects']) > 0 and 'pose' in scene_data['objects'][0]:
                        pose_data = scene_data['objects'][0]['pose']
                        transformation_matrix = np.array(pose_data)
                        file_path = os.path.join(folder_path, file_name)
                        data = np.load(file_path)
                        updated_data = {}
                        for key in data.keys():
                            if key == 'points':
                                points_array = data['points']
                                transformed_points_array = []

                                for point_pair in points_array:
                                    CP1 = point_pair[0]
                                    CP2 = point_pair[1]
                                    transformed_CP1 = transform_point(CP1, transformation_matrix)
                                    transformed_CP2 = transform_point(CP2, transformation_matrix)
                                    transformed_points_array.append([transformed_CP1, transformed_CP2])
                                updated_data['points'] = np.array(transformed_points_array)
                            elif key == 'normals':
                                normals_array = data['normals']
                                transformed_normals_array = []
                                for normal_pair in normals_array:
                                    nr1 = normal_pair[0]
                                    nr2 = normal_pair[1]
                                    transformed_nr1 = transform_normal(nr1, transformation_matrix)
                                    transformed_nr2 = transform_normal(nr2, transformation_matrix)
                                    transformed_normals_array.append([transformed_nr1, transformed_nr2])
                                updated_data['normals'] = np.array(transformed_normals_array)
                            else:
                                updated_data[key] = data[key]
                        output_file_path = os.path.join(output_folder, file_name)
                        np.savez(output_file_path, **updated_data)

if __name__ == "__main__":
    print('hi')
    # test_distance_and_coverage()
    # test_rotation_to_align_vectors()
    #test_angles()
    # test_cone_sampling()
    test_new_antipodal_grasp_sampling()
    #show_grasp_pose_definition()
    #visualise_perturbations()
    test_scoring_for_grippers()
    test_antipodal_grasp_random_sampling()
    augmented_tranformation()
    print('bye')
