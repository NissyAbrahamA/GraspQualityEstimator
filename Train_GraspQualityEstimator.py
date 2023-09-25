import os
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset
from convonets.src.checkpoints import CheckpointIO
from GraspQualityEstimator_NeuralNetwork import create_grasp_quality_net
from gag_refine.utils.transform import transform_points
from torch.optim.lr_scheduler import StepLR
import logging
import datetime
import random
from torch.optim.lr_scheduler import StepLR

hasWandB = True
try:
    import wandb
except ImportError:
    hasWandB = False

if hasWandB:
    wandb.init(project='grasp_quality_estimator')

SCENE_EDGE_LENGTH = 0.297

def pre_normalisation_tf():
    tf = torch.eye(4)
    tf[:3, :3] /= SCENE_EDGE_LENGTH  # scaling to [0, 1]
    tf[:3, 3] -= 0.5  # shifting to [-0.5, 0.5]
    return tf


def pre_normalise_points(points):
    #print('in-points')
    #print(points)
    as_numpy = False
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points)
        as_numpy = True

    tf = pre_normalisation_tf().to(points.device)
    if len(points.shape) == 3:
        tf = tf.unsqueeze(0).repeat(points.shape[0], 1, 1)
    else:
        assert len(points.shape) == 2, f'unexpected points.shape: {points.shape}'

    tf_points = transform_points(tf, points)

    if as_numpy:
        tf_points = tf_points.numpy()
    #print('transformed points')
    #print(tf_points)
    return tf_points

class CustomDataset(Dataset):
    def __init__(self, npz_folder, conv_folder):
        self.npz_files = [os.path.join(npz_folder, filename) for filename in os.listdir(npz_folder) if filename.endswith(".npz")]
        self.conv_folder = conv_folder

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        npz_file = self.npz_files[idx]
        data = np.load(npz_file)
        #print('npz_file' + str(npz_file))
        contact_points = torch.tensor(data["points"], dtype=torch.float32)
        #print(contact_points)
        contact_points = pre_normalise_points(contact_points)
        #print(contact_points.shape)
        scores = torch.tensor(data["score"], dtype=torch.float32)
        #print(scores.shape)
        # num_points = contact_points.shape[0]
        # indices = np.random.choice(num_points, size=100, replace=False)
        # contact_points = contact_points[indices]
        # scores = scores[indices]

        conv_filename = os.path.basename(npz_file).replace(".npz", ".npy")
        #print(conv_filename)
        conv_file = os.path.join(self.conv_folder, conv_filename)
        #print(conv_file)
        conv_data_dict = np.load(conv_file, allow_pickle=True)
        #print(conv_data_dict)
        conv_data_key = list(conv_data_dict.item().keys())[0]
        conv_data = conv_data_dict.item()[conv_data_key]
        #conv_data = pre_normalise_points(conv_data)
        conv_data_with_key = {
            conv_data_key: conv_data
        }

        return {
            'contact_points': contact_points,
            'scores': scores,
            'conv_data': conv_data_with_key
        }

def log_losses(loss_dict, epoch, prefix=None):
    if prefix is None:
        prefix = ''
    else:
        prefix = prefix + '/'

    for loss_name, loss_values in loss_dict.items():
        losses = np.asarray(loss_values)
        avg_loss = np.mean(losses)
        print(f'  {loss_name:15s}  avg {np.mean(losses):>7f}; min: {np.min(losses):>7f}; max: {np.max(losses):>7f}')
        logging.info(f'  {loss_name:15s}  avg {np.mean(losses):>7f}; min: {np.min(losses):>7f}; max: {np.max(losses):>7f}')
        log_dict = {f'{prefix}{loss_name}_avg_epoch': avg_loss, 'epoch': epoch}
        if hasWandB:
            wandb.log(log_dict)


def train():
<<<<<<< HEAD
    epochs = 200
    learning_rate = 0.1
    hidden_dim = 64
    loss_name = 'l1_loss'
    #loss_name = 'mse_loss'
    lr_sched_every = 100
=======
    epochs = 200
    learning_rate = 0.1
    hidden_dim = 64
    loss_name = 'l1_loss'
    #loss_name = 'mse_loss'
    lr_sched_every = 100
>>>>>>> b3712b58cb4ba656ac3deddbbc8e9907b6e17ed4

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = os.path.join(out_dir, f'log_{current_time}.txt')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    batch_size = 1
    current_dir = os.getcwd()
    train_npz_folder = os.path.join(current_dir, 'Input_NN', 'train_cpscore')
    train_conv_folder = os.path.join(current_dir, 'Input_NN', 'train_latentfeatures')
    val_npz_folder = os.path.join(current_dir, 'Input_NN', 'val_cpscore')
    val_conv_folder = os.path.join(current_dir, 'Input_NN', 'val_latentfeatures')

    train_dataset = CustomDataset(train_npz_folder, train_conv_folder)
    # for data in train_dataset:
    #     print(data)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = CustomDataset(val_npz_folder, val_conv_folder)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(f'using device: {device}')
    logging.info(f'using device: {device}')

    # grasp quality estimator
    grasp_quality_estimator = create_grasp_quality_net().to(device)
    mse_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()
    loss_fn = l1_loss
    #loss_fn = mse_loss

    optimizer = torch.optim.Adam(grasp_quality_estimator.parameters(), lr=learning_rate, weight_decay=1e-4)
<<<<<<< HEAD
    #optimizer = torch.optim.SGD(grasp_quality_estimator.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    #optimizer = torch.optim.RMSprop(grasp_quality_estimator.parameters(), lr=learning_rate, alpha=0.99, eps=1e-8, weight_decay=1e-4)
    #optimizer = torch.optim.Adagrad(grasp_quality_estimator.parameters(), lr=learning_rate, lr_decay=1e-4, weight_decay=1e-4)
    #optimizer = torch.optim.Adadelta(grasp_quality_estimator.parameters(), rho=0.9, eps=1e-6, weight_decay=1e-4)
    #optimizer = torch.optim.SGD(grasp_quality_estimator.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=1e-4)

    checkpoint_io = CheckpointIO(out_dir, model=grasp_quality_estimator, optimizer=optimizer)
    scheduler = StepLR(optimizer, step_size=lr_sched_every, gamma=0.8)
=======
    # optimizer = torch.optim.SGD(grasp_quality_estimator.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    # optimizer = torch.optim.RMSprop(grasp_quality_estimator.parameters(), lr=learning_rate, alpha=0.99, eps=1e-8,weight_decay=1e-4)
    # optimizer = torch.optim.Adagrad(grasp_quality_estimator.parameters(), lr=learning_rate, lr_decay=1e-4,weight_decay=1e-4)
    # optimizer = torch.optim.Adadelta(grasp_quality_estimator.parameters(), rho=0.9, eps=1e-6, weight_decay=1e-4)
    # optimizer = torch.optim.SGD(grasp_quality_estimator.parameters(), lr=learning_rate, momentum=0.9, nesterov=True,weight_decay=1e-4)

    checkpoint_io = CheckpointIO(out_dir, model=grasp_quality_estimator, optimizer=optimizer)
    scheduler = StepLR(optimizer, step_size=lr_sched_every, gamma=0.5)
>>>>>>> b3712b58cb4ba656ac3deddbbc8e9907b6e17ed4

    iteration = 0
    best_val_score = np.inf
    best_epoch = None
    for epoch in range(1, epochs + 1):
        grasp_quality_estimator.train()
        epoch_losses = {
            loss_name: [],
        }

        # random_indices = random.sample(range(len(train_dataset)), 1000)
        # dataloader = [train_dataset[i] for i in random_indices]

        for batch in dataloader:

            iteration += 1
            scene_encoding = batch['conv_data']
            contact_points = batch['contact_points'].float().to(device)
            #print(contact_points)
            gt_scores = batch['scores'].float().to(device)
            #print(gt_scores)
            gt_scores = gt_scores - 180.0
            gt_scores = gt_scores / 180.0
            #print(gt_scores)
            #print(gt_scores.shape)# [32,1000,1]
            gt_scores = gt_scores.squeeze(dim=2)
            #print(gt_scores.shape) #[32,1000]
            #grid_values = batch['conv_data']['grid']
            #print(grid_values.shape)[32,1,32,64,64,64]
            #print(contact_points.shape)[32,1000,2,3]
            #print(gt_scores.shape)[32,1000]
            optimizer.zero_grad()
            pred_scores = grasp_quality_estimator(contact_points, scene_encoding)

            loss = loss_fn(gt_scores, pred_scores).mean()
            loss.backward()
            optimizer.step()
            epoch_losses[loss_name].append(loss.item())
            # append_stats_to_dict(epoch_losses, gt_scores, pred_scores)

        print(f'** ep{epoch} - it{iteration}:')
        logging.info(f'** ep{epoch} - it{iteration}:')
        log_losses(epoch_losses, epoch, 'train')# log epoch instead
<<<<<<< HEAD
        optimizer.step()
        scheduler.step()

=======
        optimizer.step()
        scheduler.step()
>>>>>>> b3712b58cb4ba656ac3deddbbc8e9907b6e17ed4
        if (epoch + 1) % 10 == 0:
            # print(f'{"*" * 5} validating {"*" * 5}')
            logging.info(f'{"*" * 5} validating {"*" * 5}')
            grasp_quality_estimator.eval()
            val_losses = {
                'l1_loss': [],
                'mse_loss': [],
            }
            with torch.no_grad():
                # random_indices = random.sample(range(len(val_dataset)), 1000)
                # val_loader = [val_dataset[i] for i in random_indices]
                for val_batch in val_loader:
                    scene_encoding = val_batch['conv_data']
                    contact_points = val_batch['contact_points'].float().to(device)
                    gt_scores = val_batch['scores'].float().to(device)
                    gt_scores = gt_scores - 180.0
                    gt_scores = gt_scores / 180.0
                    gt_scores = gt_scores.squeeze(dim=2)
                    pred_scores = grasp_quality_estimator(contact_points, scene_encoding)
                    # print(gt_scores.shape)
                    # print(pred_scores.shape)
                    val_losses['mse_loss'].append(mse_loss(gt_scores, pred_scores).mean().item())
                    val_losses['l1_loss'].append(l1_loss(gt_scores, pred_scores).mean().item())
                    #append_stats_to_dict(val_losses, gt_scores, pred_scores)

            # print(f'evaluated {len(val_loader)} objects')
            logging.info(f'evaluated {len(val_loader)} objects')
            log_losses(val_losses, epoch, 'val') #epoch for logging
            val_score = np.mean(val_losses[loss_name])
            print(f'best val {loss_name} score was {best_val_score:.5f} from epoch {best_epoch}')
            logging.info(f'best val {loss_name} score was {best_val_score:.5f} from epoch {best_epoch}')
            if val_score < best_val_score:
                print('=' * 10)
                print(f'saving new best model at ep{epoch}, it{iteration}, '
                      f'with score {val_score:.5f}')
                logging.info(f'saving new best model at ep{epoch}, it{iteration}, '
                      f'with score {val_score:.5f}')
                checkpoint_io.save('model_best.pt')
                best_val_score = val_score
                best_epoch = epoch
                print('=' * 10)
            print('*' * 10)

    checkpoint_io.save('model_final.pt')
    print(f'saved grasp_quality_estimator to {os.path.join(out_dir, "model_final.pt")}')
    logging.info(f'saved grasp_quality_estimator to {os.path.join(out_dir, "model_final.pt")}')


if __name__ == '__main__':
    train()
<<<<<<< HEAD


>>>>> b3712b58cb4ba656ac3deddbbc8e9907b6e17ed4
