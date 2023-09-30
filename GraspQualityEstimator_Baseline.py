import os
import open3d as o3d
import burg_toolkit as burg
import numpy as np
import random
from scipy.spatial import cKDTree
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")

score_threshold = 350

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
testmod = True
vis_out = False
input_dir = 'C:/Users/anizy/OneDrive - Aston University/Documents/GraspQualityEstimator/input'

def refine_grasp_contact_points(ref_points, cp1, cp2):
    """
        Refines grasp contact points based on nearest neighbors in a point cloud.

        :param ref_points: numpy.ndarray
            point cloud containing graspable object surface points and normals.
        :param cp1: numpy.ndarray
            Initial contact point 1.
        :param cp2: numpy.ndarray
            Initial contact point 2.

        :return: tuple
            Returns a tuple containing refined contact points, scores, and associated data.
            - cp1: numpy.ndarray
            - cp2: numpy.ndarray
            - contact_points: numpy.ndarray
            - score: float
            - p_r: numpy.ndarray
            - q_r: numpy.ndarray
            - n_r: numpy.ndarray
            - m_r: numpy.ndarray
            - d: numpy.ndarray
        """
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

    max_score = max(grasp_contacts_list, key=lambda contact: contact[1])
    score = max_score[1]
    contact_points = max_score[0]
    return cp1, cp2, contact_points, score, p_r, q_r, n_r, m_r, d

def predict_grasp_contact_points_eval(obj_name):
    """
      Choose random grasp contact points for evaluation of grasp refinement on a given object.

      :param obj_name: str
          The name of the object for which grasp contact points are predicted.

      :return: tuple
          Returns a tuple containing evaluation results.
          - first_score: float
          - last_score: float
          - iterations: int
          - result: str
      """
    i_score = 0
    contact_points = None
    mesh_fn = os.path.join(input_dir, obj_name)
    mesh = burg.io.load_mesh(mesh_fn)
    n_sample = np.max([1500, len(mesh.triangles)])
    ref_points = burg.util.o3d_pc_to_numpy(burg.mesh_processing.poisson_disk_sampling(mesh, n_points=n_sample))
    np.random.shuffle(ref_points)

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
    # print(d)
    # print(n_r)
    # print(n_r1)
    points = np.concatenate((p_r, q_r), axis=0)
    angle_ref, angle_contact, i_score = burg.util.calc_score(d, n_r, n_r1)
    first_score = i_score
    #print('Initial score:', i_score)
    # print('Initial contact points:', points)
    iterations = 1
    if i_score > score_threshold:
        last_score = i_score
        result = 'SUCCESS'
    while i_score < score_threshold:
        cp_1, cp_2, new_contact_points, new_score, p_r, q_r, n_r, m_r, d = refine_grasp_contact_points(ref_points, cp1,cp2)
        #print(new_score)
        if new_score > i_score:
            #print('Refined score:', new_score)
            # print('Refined contact points:', new_contact_points)
            last_score = new_score
            i_score = new_score
            iterations += 1
            result = 'SUCCESS'
            cp1, cp2 = cp_1, cp_2
        else:
            last_score = new_score
            result = 'FAILURE'
            if last_score > 330:
                result = 'SUCCESS'
            break
    return first_score, last_score, iterations, result

def evaluate_grasp():
    """
       Evaluates grasp predictions for a set of objects.
    """
    results = []
    obj_name = ['002_master_chef_can.obj']
    # '029_plate.obj', '057_racquetball.obj', '063-a_marbles.obj', '065-i_cups.obj', '072-a_toy_airplane.obj'
    for obj_name in obj_name:
        for i in range(10000):
            #print(obj_name)
            print(i)
            first_score,last_score,iterations, result = predict_grasp_contact_points_eval(obj_name)

            result = {
                "Object_Name": obj_name,
                "first_score": first_score,
                "last_score": last_score,
                "iterations": iterations,
                "result": result
            }
            results.append(result)
    result_df = pd.DataFrame(results)
    result_df.to_csv('output.csv', index=False)
    print(result_df)

if __name__ == "__main__":
    gripper_model = burg.gripper.ParallelJawGripper(finger_length=0.03,
                                                    finger_thickness=0.003)
    if testmod:
        evaluate_grasp() #This is for the evaluation of the model

    if vis_out: #This is to generate the necessary visualisations of the evaluation metrics
        if not os.path.exists('Output.csv'):
            print(f"File  not found.")
        else:
            data = pd.read_csv('Output.csv')
            success_df = data[data['result'] == 'SUCCESS']
            average_success_data = success_df.groupby('Object_Name')['result'].count()
            avg_success = average_success_data.mean()
            min_first_score = data.groupby('Object_Name')['first_score'].min()
            max_first_score = data.groupby('Object_Name')['first_score'].max()
            avg_first_score = data.groupby('Object_Name')['first_score'].mean()
            avg_iterations = data['iterations'].mean()

            print('Average success for the objects out of 10000 trials: ' + str(avg_success))
            print('Average iterations for the objects out of 10000 trials: ' + str(avg_iterations))
            print('Minimum first score for the objects out of 10000 trials: ' + str(min_first_score))
            print('Maximum first score for the objects out of 10000 trials: ' + str(max_first_score))
            print('Average first score for the objects out of 10000 trials: ' + str(avg_first_score))
            avg_last_score = data.groupby('Object_Name')['last_score'].mean()
            avg_first_score = data.groupby('Object_Name')['first_score'].mean()
            average_score_improvement = avg_last_score - avg_first_score
            print('Average score improvement in all 10000 trails:(avg of last score - avg of first score) ' + str(average_score_improvement ))
            grouped_data = data.groupby(['Object_Name', 'iterations', 'result']).size().unstack(fill_value=0)
            objects = data['Object_Name'].unique()
            for obj in objects:
                obj_data = grouped_data.loc[obj]

                plt.figure(figsize=(10, 6))
                success_line, = plt.plot(obj_data.index, obj_data['SUCCESS'], marker='o', color='green', label='Success')
                failure_line, = plt.plot(obj_data.index, obj_data['FAILURE'], marker='x', linestyle='dashed', color='red',
                                         label='Failure')
                total_success_percentage = (obj_data['SUCCESS'].sum() / (
                        obj_data['SUCCESS'].sum() + obj_data['FAILURE'].sum())) * 100
                total_failure_percentage = (obj_data['FAILURE'].sum() / (
                        obj_data['SUCCESS'].sum() + obj_data['FAILURE'].sum())) * 100

                plt.xlabel('iterations')
                plt.ylabel('Number of Trials')
                plt.title(f'Success vs. Failure for {obj} - Different Numbers of Trials')

                plt.legend([f'Success (Total: {total_success_percentage:.2f}%)',
                            f'Failure (Total: {total_failure_percentage:.2f}%)'],
                           loc='best')

                plt.grid(True)
                plt.show()
