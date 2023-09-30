# GraspQualityEstimator
This repository encompasses a comprehensive project centered around two-fingered robotic grasp refinement.  It comprises two main components: a baseline model that systematically adjusts grasp contact points to improve stability and a neural network model that predicts grasp quality.

The dependencies in this projects are
burg_toolkit - (https://github.com/mrudorfer/burg-toolkit)
	The is used for dataset creation for the neural network model and for antipodal grasp sampling. Since the sample grasps need to be quantified with the Grasp score metric derived, the code for the scoring metric and necessayr visualisaiton and transformation of the dataset are all developed as part of this folder. 
	The code burg_toolkit.util.calc_score is added for the new scoring metric and it employs burg_toolkit.util.angle for calculating the grasp score.
	The code burg_toolkit.visualization.plot_contacts_normals is added for visualization of grasp 
	The code burg_toolkit.sampling.sample is modified for grasp sampling and creating dataset for the neural network model
	The code burg_toolkit.sampling.randomsample is added for random grasp generation and creating dataset for the neural network model for random poor samples
	The folder tests/gasp_testing.py functions like modified test_new_antipodal_grasp_sampling and newly added test_antipodal_grasp_random_sampling to utilise the above sampling codes of burg_toolkit.
	The folder tests/gasp_testing.py functions test_scoring_for_grippers is added for visualisation of the dataset created along with the score calculated to confirm the correctness of the dataset
	The folder tests/gasp_testing.py functions augmented_tranformation is added for centering the YCB dataset to make them consistent with ConVSDFNet for latent code representation.

latent_representation is generated using gag_refine and convonets
	data folder -input for latent code generation
	latent_representation/convonets/src/conv_sdfnet/interface.py eval_scene_pc_for_latent_code function is added and  latent_representation/scripts/optimise_to_surface.py get_latent_code is modified to generate the latent codes of the YCB dataset. 
	

The important components of Baseline Model are
	GraspQualityEstimator_Baseline.py  - The experimentaiton with random grasps chosen for refinement for 6 YCB object implemenation 
	GraspQualityEstimator_Baseline_Demo.py - Developed for Demo to generate success grasp
	input - Input object pointcloud for verification of the baseline model
	out_from_baseline folder - Generated indidual output and visualisation of the Baseline model
	Output.xls file - Input for visualisaiton generation of the baseline model with consolidated output of baseline model.
	
	
The important componenets for Neural Network model are
	GraspQualityEstimator_NeuralNetwork.py
	Train_GraspQualityEstimator.py
	Test_GraspQualityEstimator.py
	Input_NN - The train,val and test dataset with latent features and the grasps with quality score for training and tesing of the neural network model
	out - the output logs and the best_model after trianing of the model is stored in this folder
	out_test- out logs of the testing of the neural network.
	
test - different visualisation and testing of the datasets.
071_a_toy_airplane and 011_banana are visualisation of latent space
