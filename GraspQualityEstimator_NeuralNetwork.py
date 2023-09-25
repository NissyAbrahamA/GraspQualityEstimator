import torch
import torch.nn as nn
import torch.nn.functional as F
from convonets.src.checkpoints import CheckpointIO
from convonets.src.layers import ResnetBlockFC
from convonets.src.common import normalize_coordinate, normalize_3d_coordinate
import numpy as np
def create_grasp_quality_net():
    grasp_quality_estimator = GraspQualityEstimator()
    #print(grasp_quality_estimator)
    return grasp_quality_estimator

# class SelfAttention(nn.Module):
#     def __init__(self, input_dim, num_heads):
#         super(SelfAttention, self).__init__()
#         self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
#
#     def forward(self, x):
#         x, _ = self.attention(x, x, x)
#         return x

# def load_grasp_quality_net(model_fn='model.pt'):
#     grasp_quality_net = create_grasp_quality_net()
#     checkpoint_io = CheckpointIO(config['training']['out_dir'], model=grasp_quality_net)
#     checkpoint_io.load(model_fn)
#      return grasp_quality_net


class GraspQualityEstimator(nn.Module):
    """
    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        pooling (str): pooling used, max|avg
    """

    def __init__(self, dim=3, c_dim=32, hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1,
                 pooling='max'):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.dim = dim
        self.fc_p = nn.Linear(dim, hidden_size)  # position encoder
        self.fc_c = nn.ModuleList([
            nn.Linear(c_dim, hidden_size) for _ in range(n_blocks)
        ])
        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size),
            *[ResnetBlockFC(2 * hidden_size, hidden_size) for _ in range(n_blocks - 1)]
        ])
        assert pooling in ['max', 'avg'], f'pooling is {pooling}'
        if pooling == 'max':
            self.pool = lambda x: torch.max(x, dim=-2, keepdim=True)[0]
        else:
            self.pool = lambda x: torch.mean(x, dim=-2, keepdim=True)

        self.actvn = F.relu if not leaky else lambda x: F.leaky_relu(x, 0.2)
        self.fc_out = nn.Linear(hidden_size, 1)

        self.sample_mode = sample_mode
        self.padding = padding
        #self.self_attention = SelfAttention(hidden_size, num_heads=4)

    # def sample_plane_feature(self, p, c, plane='xz'):
    #     xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)  # normalize to the range of (0, 1)
    #     xy = xy[:, :, None].float()
    #     vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
    #     c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
    #     return c

    # def sample_grid_feature(self, p, c):
    #     print('p')#torch.Size([1, 4760, 3])
    #     print(p.shape)
    #     print('c')#torch.Size([1, 1, 32, 64, 64, 64])
    #     print(c.shape)
    #     p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)  # normalize to the range of (0, 1)
    #     p_nor = p_nor[:, :, None, :].float()
    #     print(p_nor.shape)#torch.Size([1, 4760, 1, 1, 3])
    #     vgrid = 2.0 * p_nor - 1.0  # Select only the x and y coordinates
    #     vgrid = vgrid.unsqueeze(3)
    #     print(vgrid.shape)#torch.Size([1, 4760, 1,  2])
    #     c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
    #     return c

    def sample_grid_feature(self, p, c):
        #print(c.shape)
        c = c.squeeze(1) # confirm
        #print(c.shape)
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)  # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        #print(p_nor.shape)
        vgrid = 2.0 * p_nor - 1.0  # normalize to (-1, 1)
        #print(vgrid.shape)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze \
            (-1).squeeze(-1)
        return c

    def forward(self, p, c_plane, **kwargs):
        # gathering the features only works with a 3-dim tensor, therefore we need to reshape
        # p is in the form of: [b1, b2, n, 3], if not, we have to adjust it to b1 of c
        # c is in the form of: [b1, c_dim, plane_res, plane_res]
        #   b1 = number of scenes
        #   b2 = number of grasps
        #   n = number of contact points
        squeeze = False
        if len(p.shape) == 3:
            squeeze = True
            p = p.unsqueeze(0)
        #print(p.shape)
        #print(p)
        b1, b2, n = p.shape[0], p.shape[1], p.shape[2]
        # print(b1)#1
        # print(b2)#2553
        # print(n)#2
        p = p.reshape(b1, n*b2, self.dim)

        plane_type = list(c_plane.keys())
        c = 0
        #print(p)
        #grid_tensor = c_plane['grid']
        #grid_tensor_cpu = grid_tensor.cpu().numpy()
        #np.save('c_plane.npy',grid_tensor_cpu)
        if 'grid' in plane_type:
            c += self.sample_grid_feature(p, c_plane['grid'])
        if 'xz' in plane_type:
            c += self.sample_plane_feature(p, c_plane['xz'], plane='xz')
        if 'xy' in plane_type:
            c += self.sample_plane_feature(p, c_plane['xy'], plane='xy')
        if 'yz' in plane_type:
            c += self.sample_plane_feature(p, c_plane['yz'], plane='yz')

        c = c.transpose(1, 2)
        #print(c.shape)#torch.Size([1, 5106,32])
        #ft_tensor = c
        #ft_tensor_cpu = ft_tensor.cpu().numpy()
        #np.save('features.npy',ft_tensor_cpu)

        # reshape back
        p = p.reshape(b1, b2, n, self.dim)
        c = c.reshape(b1, b2, n, self.c_dim)

        p = p.float()
        net = self.fc_p(p)  # first layer uses only positional encoding, no latent code
        for i, block in enumerate(self.blocks):
            net = net + self.fc_c[i](c)  # add processed features from scene encoding
            # concatenate pooled features only after first block
            if i > 0:
                pooled = self.pool(net)  # (b1, b2, 1, hidden_dim)
                pooled = pooled.repeat(1, 1, n, 1)  # propagate to the number of contact points
                net = torch.cat([net, pooled], dim=-1)  # (b1, b2, n, 2*hidden_dim)
            net = block(net)  # (b1, b2, n, hidden_dim)

        net = self.pool(net).squeeze(-2)  # final pool, then reduce contact point dimension -> (b1, b2, hidden_dim)
        #net = self.self_attention(net)
        out = self.fc_out(self.actvn(net))  # final layer
        out = out.squeeze(-1)  # squeeze to (b1, b2), we have one predicted value per grasp
        if squeeze:
            out = out.squeeze(0)
        out = torch.tanh(out)
        return out

#consider median loss
#mse training
#try sigmoid activation