import os.path
import torch
from GraspQualityEstimator_PointNet import PointNetFeat, GraspScorePredictor, CustomDataset
from torch.utils.data import DataLoader

loaded_pointnetpp_feature_model = PointNetFeat()
loaded_grasp_score_predictor = GraspScorePredictor()

loaded_pointnetpp_feature_model.load_state_dict(torch.load('pointnet_feature_model.pth'))
loaded_grasp_score_predictor.load_state_dict(torch.load('grasp_score_predictor.pth'))

loaded_pointnetpp_feature_model.eval()
loaded_grasp_score_predictor.eval()


batch_size = 10  #check
npz_dir = "C://Users//anizy//OneDrive - Aston University//Documents//GraspQualityEstimator//input_npz//test"
test_npz_files = [os.path.join(npz_dir, filename) for filename in os.listdir(npz_dir) if filename.endswith(".npz")]
test_conv_data_dir = "C://Users//anizy//OneDrive - Aston University//Documents//GraspQualityEstimator//input_conv//test"

test_dataset = CustomDataset(test_npz_files, test_conv_data_dir)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
criterion = torch.nn.MSELoss()

total_loss = 0.0
with torch.no_grad():
    for batch_idx, (contact_points, scores, conv_data) in enumerate(test_dataloader):
        learned_features = loaded_pointnetpp_feature_model(conv_data)
        combined_features = torch.cat([learned_features, contact_points, scores], dim=1)
        predicted_scores = loaded_grasp_score_predictor(combined_features)
        #predicted_score = predicted_scores.item()
        batch_loss = criterion(predicted_scores, scores)
        total_loss += batch_loss.item()

#print(f"Predicted Grasp Score: {predicted_score:.4f}")
average_loss = total_loss / len(test_dataloader)
print(f"Average Test Loss: {average_loss:.4f}")