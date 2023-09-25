import torch
import numpy as np
import yaml
import open3d as o3d


def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config


def main():
    # config_path = "path/to/your/config.yaml"
    checkpoint_path = "model_best.pt"
    config = load_config('config.yaml')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model = checkpoint['model']
    #model.eval()

    point_cloud = o3d.io.read_point_cloud("002_master_chef_can_pc.npy")
    input_point_cloud = np.asarray(point_cloud.points).astype(np.float32)

    input_tensor = torch.tensor(input_point_cloud, dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor.unsqueeze(0))
        np.save("output.npy", output.numpy())
        print("Output shape:", output.shape)


if __name__ == "__main__":
    main()
