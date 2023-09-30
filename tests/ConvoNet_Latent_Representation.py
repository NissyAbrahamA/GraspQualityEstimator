import argparse
from timeit import default_timer as timer
import pprint

import torch
import numpy as np
import burg_toolkit as burg
import trimesh

from convonets.src import config
from convonets.src.common import sdf_to_occ


def parse_args():
    parser = argparse.ArgumentParser(
        description='refine the gripper pose and configuration in a given scene'
    )
    parser.add_argument('convonet_config', type=str, help='path to ConvONet config file')

    return parser.parse_args()


def main(args):

    cfg = config.load_config(args.convonet_config, config.default_config_fn)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset
    dataset = config.get_dataset('test', cfg, return_idx=True)
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=0, shuffle=False)

    backbone = config.get_model_interface(cfg, device, dataset)

    for it, data in enumerate(test_loader):
        obj_idx = data['idx'].item()
        print(f'{"*"*5} object {obj_idx}')

        # process scene point cloud to create latent representation
        latent_code = backbone.eval_scene_pointcloud(data)
        np.save('latent_codes.npy', latent_code)


if __name__ == '__main__':
    main(parse_args())
