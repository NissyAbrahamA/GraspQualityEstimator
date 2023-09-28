import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset
from convonets.src.checkpoints import CheckpointIO
from GraspQualityEstimator_NeuralNetwork import create_grasp_quality_net
from Train_GraspQualityEstimator import CustomDataset, log_losses
import logging
import datetime
#
# hasWandB = True
# try:
#     import wandb
# except ImportError:
#     hasWandB = False
#
# if hasWandB:
#     wandb.init(project='grasp_quality_estimator_')

def test(test_npz_folder, test_conv_folder, model_path):
    batch_size = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = create_grasp_quality_net().to(device)
    checkpoint = torch.load(model_path)
    #print(checkpoint.keys())
    model.load_state_dict(checkpoint['model'])
    model.eval()

    test_dataset = CustomDataset(test_npz_folder, test_conv_folder)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_losses = {
        'l1_loss': [],
        'mse_loss': [],
    }

    with torch.no_grad():
        for idx, batch in enumerate(test_loader, 1):
            scene_encoding = batch['conv_data']
            contact_points = batch['contact_points'].float().to(device)
            gt_scores = batch['scores'].float().to(device)
            gt_scores = gt_scores - 180.0
            gt_scores = gt_scores / 180.0
            gt_scores = gt_scores.squeeze(dim=2)
            pred_scores = model(contact_points, scene_encoding)
            # print(gt_scores.shape)
            # print(pred_scores.shape)
            mse = torch.nn.MSELoss()(gt_scores, pred_scores).mean().item()
            l1 = torch.nn.L1Loss()(gt_scores, pred_scores).mean().item()

            test_losses['mse_loss'].append(mse)
            test_losses['l1_loss'].append(l1)

            logging.info(f'Test Batch {idx}/{len(test_loader)}: MSE Loss: {mse:.5f}, L1 Loss: {l1:.5f}')

    avg_mse_loss = np.mean(test_losses['mse_loss'])
    avg_l1_loss = np.mean(test_losses['l1_loss'])

    print(f'Test MSE Loss (Average): {avg_mse_loss:.5f}')
    print(f'Test L1 Loss (Average): {avg_l1_loss:.5f}')

    logging.info(f'Test MSE Loss (Average): {avg_mse_loss:.5f}')
    logging.info(f'Test L1 Loss (Average): {avg_l1_loss:.5f}')

if __name__ == '__main__':
    current_dir = os.getcwd()
    test_npz_folder = os.path.join(current_dir, 'Input_NN', 'val_cpscore')
    test_conv_folder = os.path.join(current_dir, 'Input_NN', 'val_latentfeatures')
    model_path = os.path.join('out', 'model_final.pt')

    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = os.path.join('out', f'test_log_{current_time}.txt')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    test(test_npz_folder, test_conv_folder, model_path)
