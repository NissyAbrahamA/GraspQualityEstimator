import argparse
import os
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA


def parse_args():
    parser = argparse.ArgumentParser(
        description='visualise the latent codes'
    )
    parser.add_argument('--generation_dir', type=str, help='path to generation dir',
                        default='../out/shapenetv2_grid64_fullpc_sdf_bs16_lrsched/generation/')
    parser.add_argument('--shape', type=str, help='shape id',
                        default='d7ec18a98781e5a444175b4dddf5be08')

    return parser.parse_args()


def main(z):
    #print(z)
    print(f'got data: {z.shape}, max {z.max()}, min {z.min()}, mean {z.mean()}')
    latent_dim = z.shape[1]
    n_grid = z.shape[2]
    assert np.allclose(z.shape[2:], n_grid)

    pca = PCA(n_components=1)
    batch_size, channels, spatial_dim = z.size(0), z.size(1), z.size()[2:]
    z_reshaped = z.view(batch_size, channels, -1).transpose(1, 2).reshape(-1, channels)
    z_reshaped = z_reshaped.cpu().numpy()
    X = pca.fit_transform(z_reshaped.reshape(-1, latent_dim))  # to (64, 64, 64, 32)
    X = np.ascontiguousarray(X).reshape((n_grid, n_grid, n_grid))
    print(f'transformed with PCA into: {X.shape}, max {X.max()}, min {X.min()}, mean {X.mean()}')

    #plot_heatmap(X)
    #show_occupancy(X)
    animate_heatmap(X)

def plot_heatmap(X, n_dim=64, y_offset=32, slice_thickness=2):
    xmax = X.max()
    # get y-direction indices and apply mask according to offset
    print(np.indices(X.shape).shape)
    y = np.indices(X.shape)[1]
    mask = np.bitwise_and(y >= y_offset, y < y_offset + slice_thickness)
    print('y', y.shape)
    print('mask', mask.shape)
    y = y[mask]
    print('y_masked', y.shape)
    x = np.indices(X.shape)[0][mask]
    z = np.indices(X.shape)[2][mask]
    col = X[mask].flatten() / xmax

    # 3D Plot
    fig = plt.figure(figsize=(8, 8))
    ax3D = fig.add_subplot(projection='3d')
    p3d = ax3D.scatter(x, y, z, c=col, cmap=plt.colormaps['hsv'])
    fig.colorbar(p3d)

    ax3D.set_xlim([0, n_dim+1])
    ax3D.set_ylim([0, n_dim+1])
    ax3D.set_zlim([0, n_dim+1])
    # ax3D.set_proj_type('ortho')
    ax3D.set_aspect('equalxz')

    ax3D.set_xlabel('X')
    ax3D.set_ylabel('Y')
    ax3D.set_zlabel('Z')
    save_path = '/content/drive/MyDrive/dev/gripper-aware-grasp-refinement/scripts/3d_plot.png'
    plt.show()
    plt.savefig(save_path)
    print('saved to ' + save_path)


def animate_heatmap(X, n_dim=64, save=True):
    xmax = X.max()
    y = np.indices(X.shape)[1]
    x = np.indices(X.shape)[0]
    z = np.indices(X.shape)[2]

    # 3D Plot
    fig = plt.figure(figsize=(8, 8))
    ax3D = fig.add_subplot(projection='3d')
    cmap = plt.colormaps['hsv']
    p3d = ax3D.scatter([], [], [], c=[], cmap=cmap)
    fig.colorbar(p3d)

    ax3D.set_xlim([0, n_dim+1]), ax3D.set_ylim([0, n_dim+1]), ax3D.set_zlim([0, n_dim+1])
    ax3D.set_xlabel('X'), ax3D.set_ylabel('Y'), ax3D.set_zlabel('Z')
    ax3D.set_aspect('equalxz')
    ax3D.set_title(args.shape)

    def update(y_idx):
        mask = y == y_idx
        p3d._offsets3d = (x[mask], y[mask], z[mask])
        p3d.set_color(cmap(X[mask].flatten() / xmax))

    frames = np.hstack([np.arange(n_dim), np.arange(n_dim-2, 0, -1)])
    animation = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=60
    )
    plt.show()

    if save:
        save_path = os.path.join('/content/drive/MyDrive/dev/gripper-aware-grasp-refinement/scripts/', f'{args.shape}.gif')
        animation.save(save_path)


def show_occupancy(X):
    indices = np.where(X > 1)
    xs = indices[0]
    ys = indices[1]
    zs = indices[2]

    fig = plt.figure(figsize=(16, 16))
    ax3D = fig.add_subplot(projection='3d')
    p3d = ax3D.scatter(xs, ys, zs)

    ax3D.set_xlabel('X')
    ax3D.set_ylabel('Y')
    ax3D.set_zlabel('Z')

    plt.show()


if __name__ == '__main__':
    args = parse_args()
    #latent_code_fn = os.path.join(args.generation_dir, 'latent_codes', f'{args.shape}.npy')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = "072-a_toy_airplane.npy"
    full_path = os.path.join(script_dir, filename)
    data = np.load(full_path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.ndim == 0:
        # Directly access the scalar value
        scalar_value = data.item()
        # Assuming 'scalar_value' is a dictionary, you can access its elements
        if 'grid' in scalar_value:
            data = scalar_value['grid']

        else:
            print("No 'grid' key found in the dictionary.")
    else:
        print("Data is not a scalar NumPy array.")
    main(data)
