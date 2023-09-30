import os.path
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, npz_files, conv_data_dir):
        self.npz_files = npz_files
        self.conv_data_dir = conv_data_dir

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        npz_file = self.npz_files[idx]
        data = np.load(npz_file)
        contact_points = torch.tensor(data["points"], dtype=torch.float32)
        scores = torch.tensor(data["score"], dtype=torch.float32)

        conv_data = os.path.join(self.conv_data_dir, f"conv_{idx}.npy")
        conv_data = torch.tensor(np.load(conv_data), dtype=torch.float32)

        return contact_points, scores, conv_data


class PointNetFeat(nn.Module):
    def __init__(self, input_channels):
        super(PointNetFeat, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2)[0]
        return x
#modify to use gradient to predict good grasp
#check pointnet1 implementation
class GraspScorePredictor(nn.Module):
    def __init__(self, input_dim):
        super(GraspScorePredictor, self).__init__()

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, combined_features):

        x = self.fc1(combined_features)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        predicted_scores = self.fc3(x)# apply sigmoid layer after this (0,1)*360
        predicted_scores = torch.sigmoid(predicted_scores)
        predicted_scores = predicted_scores * 360
        return predicted_scores

npz_dir = "C://Users//anizy//OneDrive - Aston University//Documents//GraspQualityEstimator//input_npz//train"
npz_files = [os.path.join(npz_dir, filename) for filename in os.listdir(npz_dir) if filename.endswith(".npz")]
conv_data_dir = "C://Users//anizy//OneDrive - Aston University//Documents//GraspQualityEstimator//input_conv//train"
dataset = CustomDataset(npz_files, conv_data_dir)
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

pointnet_feature_model = PointNetFeat()
num_epochs = 100
grasp_score_predictor = GraspScorePredictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(grasp_score_predictor.parameters(), lr=0.001)
for epoch in range(num_epochs):
    for batch_idx, (contact_points, scores, conv_data) in enumerate(dataloader):
            optimizer.zero_grad()
            learned_features = pointnet_feature_model(conv_data)
            combined_features = torch.cat([learned_features, contact_points, scores], dim=1)
            predicted_scores = grasp_score_predictor(combined_features)
            loss = criterion(predicted_scores, scores)
            loss.backward()
            optimizer.step()


            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")


torch.save(pointnet_feature_model.state_dict(), 'pointnet_feature_model.pth')
torch.save(grasp_score_predictor.state_dict(), 'grasp_score_predictor.pth')

# #load the models to test  - see reference in notes