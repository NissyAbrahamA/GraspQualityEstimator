# GraspQualityEstimator

This repository encompasses a comprehensive project centered around two-fingered robotic grasp refinement. It comprises two main components: a baseline model that systematically adjusts grasp contact points to improve stability and a neural network model that predicts grasp quality.

## Build Environment

To replicate the development environment for this project, you can use Conda. Follow these steps:
-  conda create -n gqe python=3.10
- conda activate gqe
- conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
- conda config --append channels conda-forge
- conda install cython pandas pyembree
- pip install https://github.com/mrudorfer/burg-toolkit/archive/refs/heads/dev.zip
- pip install pykdtree plyfile tensorboardX gdown wandb
- pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
- python setup.py build_ext --inplace
- pip install -e .


## Dependencies

The dependencies in this project are:

### burg_toolkit
- [GitHub Repository](https://github.com/mrudorfer/burg-toolkit)
  
The `burg_toolkit` is used for dataset creation for the neural network model and for antipodal grasp sampling. It includes code for scoring metrics, visualization, and dataset transformation. Notable modifications and additions include:

- `burg_toolkit.util.calc_score`: New scoring metric.
- `burg_toolkit.util.angle`: Used for calculating grasp scores.
- `burg_toolkit.visualization.plot_contacts_normals`: Added for grasp visualization.
- `burg_toolkit.sampling.sample`: Modified for grasp sampling.
- `burg_toolkit.sampling.randomsample`: Added for random grasp generation.
- `tests/gasp_testing.py`:  functions like modified test_new_antipodal_grasp_sampling and newly added test_antipodal_grasp_random_sampling to utilise the above sampling codes of burg_toolkit.
- `tests/gasp_testing.py`: functions test_scoring_for_grippers is added for visualisation of the dataset created along with the score calculated to confirm the correctness of the dataset
- `tests/gasp_testing.py`: functions augmented_tranformation is added for centering the YCB dataset to make them consistent with ConVSDFNet for latent code representation.

### Latent Representation

Latent representation is generated using `gag_refine` and convolutional networks. Key components include:

- `data` folder: Input for latent code generation.
- `latent_representation/convonets/src/conv_sdfnet/interface.py`: Added `eval_scene_pc_for_latent_code` function.
- `latent_representation/scripts/optimise_to_surface.py`: Modified `get_latent_code` for generating latent codes of the YCB dataset.

## Baseline Model

The important components of the Baseline Model are:

- `GraspQualityEstimator_Baseline.py`: Experimentation with random grasps chosen for refinement for 6 YCB objects.
- `GraspQualityEstimator_Baseline_Demo.py`: Developed for demo to generate successful grasps.
- `input`: Input object point cloud for verification of the baseline model.
- `out_from_baseline` folder: Generated individual output and visualization of the Baseline model.
- `Output.xls` file: Input for visualization generation of the baseline model with consolidated output.

## Neural Network Model

The important components for the Neural Network model are:

- `GraspQualityEstimator_NeuralNetwork.py`
- `Train_GraspQualityEstimator.py`
- `Test_GraspQualityEstimator.py`
- `Input_NN`: The train, validation, and test dataset with latent features and grasps with quality scores for training and testing the neural network model.
- `out`: Output logs and the best model after training of the model are stored in this folder.
- `out_test`: Output logs of the testing of the neural network.

Various visualizations and testing of the datasets are avialable in Test folder, including `071_a_toy_airplane` and `011_banana` for visualization of the latent space.

## Testing Baseline Model

The GraspQualityEstimator_Baseline.py can be run on CPU machines without any issues since it contains a hardcoded experimental setup.
Similarly, the GraspQualityEstimator Baseline Demo.py is also compatible with CPU machines. This program aims to identify successful
grasps through an iterative refinement process.

## Testing Neural Network Model 
After setting up the environment as explained above, the Test_GraspQualityEstimator.py script can be executed on a GPU machine.
This script is designed to be run on the testing dataset located in the Input_NN folder
