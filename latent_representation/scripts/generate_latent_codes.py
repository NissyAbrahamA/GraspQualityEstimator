import argparse
import os

import torch
import numpy as np
import burg_toolkit as burg

from convonets.src import config


def parse_args():
    parser = argparse.ArgumentParser(
        description='save the latent codes to generation file'
    )
    parser.add_argument('convonet_config', type=str, help='path to ConvONet config file')

    return parser.parse_args()


def main(args):
    cfg = config.load_config(args.convonet_config, config.default_config_fn)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # specify dirs where to put the latent codes
    out_dir = cfg['training']['out_dir']
    latent_codes_dir = os.path.join(out_dir, cfg['generation']['generation_dir'], 'latent_codes')
    burg.io.make_sure_directory_exists(latent_codes_dir)

    # load dataset
    dataset = config.get_dataset('test', cfg, return_idx=True)
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=0, shuffle=False)

    # load ConvONet model and checkpoint
    convonet = config.get_model_interface(cfg, device, dataset)

    for it, data in enumerate(test_loader):
        scene_idx = data['idx'].item()
        model_dict = dataset.get_model_dict(scene_idx)
        model_name = model_dict['model']
        print(f'{it}/{len(test_loader)}: {"*"*5} scene {scene_idx} - {model_name}')

        # process scene point cloud to create latent representation
        convonet.eval_scene_pointcloud(data)
        z = convonet.latent_code
        fn = os.path.join(latent_codes_dir, f'{model_name}.npy')
        np.save(fn, z['grid'].cpu().numpy())


if __name__ == '__main__':
    main(parse_args())
