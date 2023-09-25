import os
import open3d as o3d
import burg_toolkit as burg
import numpy as np
import random
from scipy.spatial import cKDTree
import itertools
import pandas as pd

score_threshold = 330
total_random_points_chosen = 0
total_successful_refinements = 0
total_first_attempt_success = 0

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
testmod = False


def refine_grasp_contact_points(ref_points, cp1, cp2):
    kdtree = cKDTree(ref_points)
    distance_threshold = 0.01
    new_cp1 = kdtree.query_ball_point(cp1, r=distance_threshold)
    new_cp2 = kdtree.query_ball_point(cp2, r=distance_threshold)
    grasp_contacts_list = []
    for point1_idx, point2_idx in itertools.product(new_cp1, new_cp2):
        cp1 = ref_points[point1_idx]
        cp2 = ref_points[point2_idx]
        p_r = ref_points[point1_idx, 0:3]
        q_r = ref_points[point2_idx, 0:3]
        n_r = ref_points[point1_idx, 3:6]
        m_r = ref_points[point2_idx, 3:6]
        d = (q_r - p_r).reshape(-1, 3)
        angle_ref, angle_contact, score = burg.util.calc_score(d, n_r, m_r)

        points = np.concatenate((p_r, q_r), axis=0)
        grasp_contacts_list.append((points, score))
    # for point1_idx in new_cp1:#see
    #     for point2_idx in new_cp2:#try vectorisation
    #         p_r = ref_points[point1_idx, 0:3]
    #         q_r = ref_points[point2_idx, 0:3]
    #         n_r = ref_points[point1_idx, 3:6]
    #         m_r = ref_points[point2_idx, 3:6]
    #
    #         d = (q_r - p_r).reshape(-1, 3)
    #         #print(d)
    #         angle_ref, angle_contact, score = burg.util.calc_score(d, n_r, m_r)
    #
    #         points = np.concatenate((p_r, q_r), axis=0)
    #         grasp_contacts_list.append((points, score))
    max_score = max(grasp_contacts_list, key=lambda contact: contact[1])
    score = max_score[1]
    contact_points = max_score[0]
    return cp1, cp2, contact_points, score, p_r, q_r, n_r, m_r, d


def predict_grasp_contact_points(obj_name):
    global total_random_points_chosen, total_successful_refinements, total_first_attempt_success
    gripper_model = burg.gripper.ParallelJawGripper(finger_length=0.03,
                                                    finger_thickness=0.003)
    input_dir = 'C:/Users/anizy/OneDrive - Aston University/Documents/GraspQualityEstimator/input'

    i_score = 0
    contact_points = None
    successful_refinements = 0
    total_attempts = 0
    random_points_chosen = 0
    first_attempt_success = False

    mesh_fn = os.path.join(input_dir, obj_name)
    mesh = burg.io.load_mesh(mesh_fn)
    n_sample = np.max([1500, len(mesh.triangles)])
    ref_points = burg.util.o3d_pc_to_numpy(burg.mesh_processing.poisson_disk_sampling(mesh, n_points=n_sample))
    np.random.shuffle(ref_points)

    max_attempts = 1000
    attempts = 0

    index1 = random.randint(0, len(ref_points) - 1)
    index2 = random.randint(0, len(ref_points) - 1)
    while index2 == index1:
        index2 = random.randint(0, len(ref_points) - 1)
    cp1 = ref_points[index1]
    p_r = ref_points[index1, 0:3]
    n_r = ref_points[index1, 3:6]
    cp2 = ref_points[index2]
    q_r = ref_points[index2, 0:3]
    n_r1 = ref_points[index2, 3:6]
    d = (q_r - p_r).reshape(-1, 3)
    points = np.concatenate((p_r, q_r), axis=0)
    angle_ref, angle_contact, i_score = burg.util.calc_score(d, n_r, n_r1)

    print('Initial score:', i_score)
    print('Initial contact points:', points)
    if not testmod:
        burg.visualization.plot_contacts_normals(mesh, p_r, q_r, n_r, n_r1, i_score)
    # print(cp1)
    # print(cp2)

    while i_score < score_threshold and attempts <= max_attempts:
        # print(i_score)
        cp_1, cp_2, new_contact_points, new_score, p_r, q_r, n_r, m_r, d = refine_grasp_contact_points(ref_points, cp1,
                                                                                                       cp2)
        if new_score > i_score:
            print('Refined score:', new_score)
            print('Refined contact points:', new_contact_points)

            if not testmod:
                burg.visualization.plot_contacts_normals(mesh, p_r, q_r, n_r, m_r, new_score)
            i_score = new_score
            cp1, cp2 = cp_1, cp_2

            successful_refinements += 1
            if i_score >= score_threshold:
                if attempts == 1:
                    first_attempt_success = True
            total_successful_refinements += 1
        else:
            new_index1 = random.randint(0, len(ref_points) - 1)
            new_index2 = random.randint(0, len(ref_points) - 1)
            while new_index2 == new_index1:
                new_index2 = random.randint(0, len(ref_points) - 1)
            new_cp1 = ref_points[new_index1]
            new_p_r = ref_points[new_index1, 0:3]
            new_n_r = ref_points[new_index1, 3:6]
            new_cp2 = ref_points[new_index2]
            new_q_r = ref_points[new_index2, 0:3]
            new_n_r1 = ref_points[new_index2, 3:6]
            new_d = (new_q_r - new_p_r).reshape(-1, 3)
            new_points = np.concatenate((new_p_r, new_q_r), axis=0)
            new_angle_ref, new_angle_contact, new_score = burg.util.calc_score(new_d, new_n_r, new_n_r1)
            random_points_chosen += 1
            total_random_points_chosen += 1
            # print('new_i_score' + str(new_score))
            # if new_score > i_score:
            # print('here')
            attempts = 0
            cp1 = new_cp1
            p_r = new_p_r
            n_r = new_n_r
            cp2 = new_cp2
            q_r = new_q_r
            n_r1 = new_n_r1
            d = new_d
            points = new_points
            i_score = new_score

        if i_score >= score_threshold:
            contact_points = np.concatenate((p_r, q_r), axis=0)
            if not testmod:
                burg.visualization.plot_contacts_normals(mesh, p_r, q_r, n_r, n_r1, i_score)
            print('The contact points for a good grasp are ' + str(contact_points))
            print('The grasp quality score is ' + str(i_score))
            break

        attempts += 1
        total_attempts += 1
    if first_attempt_success:
        total_first_attempt_success += 1
    random_points_chosen = 0
    successful_refinements = 0
    first_attempt_success = False

    if attempts >= max_attempts:
        print('Maximum attempts reached. Could not find better contact points. ')


def evaluate_grasp():
    global total_random_points_chosen, total_successful_refinements, total_first_attempt_success
    results = []
    obj_name = ['002_master_chef_can.obj', '003_cracker_box.obj', '011_banana.obj', '018_plum.obj', '025_mug.obj',
                '029_plate.obj', '057_racquetball.obj', '063-a_marbles.obj', '065-i_cups.obj', '072-a_toy_airplane.obj']
    for obj_name in obj_name:
        for _ in range(100):
            predict_grasp_contact_points(obj_name)

        average_random_points_chosen = total_random_points_chosen / 100
        average_successful_refinements = total_successful_refinements / 100
        average_first_attempt_success = total_first_attempt_success / 100

        # print(f"Total first attempt success (out of 100 attempts): {total_first_attempt_success}")
        # print(f"Total random points chosen (out of 100 attempts): {total_random_points_chosen}")
        # print(f"Total successful refinements for good grasp (out of 100 attempts): {total_successful_refinements}")
        # print(f"Average first attempt success (out of 100 attempts): {average_first_attempt_success}")
        # print(f"Average random points chosen (out of 100 attempts): {average_random_points_chosen}")
        # print(f"Average successful refinements for good grasp (out of 100 attempts): {average_successful_refinements}")

        result = {
            "Object Name": obj_name,
            "Total First Attempt Success ( out of 100 attempts)": total_first_attempt_success,
            "Total Random Points Chosen ( out of 100 attempts)": total_random_points_chosen,
            "Total Successful Refinements ( out of 100 attempts)": total_successful_refinements,
            "Average First Attempt Success": average_first_attempt_success,
            "Average Random Points Chosen": average_random_points_chosen,
            "Average Successful Refinements": average_successful_refinements
        }
        results.append(result)
        total_random_points_chosen = 0
        total_successful_refinements = 0
        total_first_attempt_success = 0
    result_df = pd.DataFrame(results)
    result_df.to_csv('output.csv', index=False)

    print(result_df.to_string(index=False))


def get_grasp_for_object():
    obj_name = "002_master_chef_can.obj"
    predict_grasp_contact_points(obj_name)


if __name__ == "__main__":
    if testmod:
        evaluate_grasp()
    else:
        get_grasp_for_object()
