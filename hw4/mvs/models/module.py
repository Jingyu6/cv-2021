import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        
        def build_conv_bn_relu(in_c, out_c, kernel_size=3, stride=1, padding=1):
            conv = nn.Conv2d(
                in_channels=in_c, 
                out_channels=out_c, 
                kernel_size=kernel_size, 
                stride=stride,
                padding=padding
            )
            bn = nn.BatchNorm2d(num_features=out_c)
            relu = nn.ReLU()
            return [conv, bn, relu]

        self.layers = \
            build_conv_bn_relu(3, 8) + \
            build_conv_bn_relu(8, 8) + \
            build_conv_bn_relu(8, 16, kernel_size=5, stride=2, padding=2) + \
            build_conv_bn_relu(16, 16) + \
            build_conv_bn_relu(16, 16) + \
            build_conv_bn_relu(16, 32, kernel_size=5, stride=2, padding=2) + \
            build_conv_bn_relu(32, 32) + \
            build_conv_bn_relu(32, 32) + \
            [nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        # x: [B,3,H,W]
        return self.layers(x)


class SimlarityRegNet(nn.Module):
    def __init__(self, G):
        super(SimlarityRegNet, self).__init__()
        
        self.conv1 = nn.Conv2d(G, 8, 3, 1, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 2, 1)
        self.conv3 = nn.Conv2d(16, 32, 3, 2, 1)
        
        self.conv_trans1 = nn.ConvTranspose2d(32, 16, 3, 2, 1, 1)
        self.conv_trans2 = nn.ConvTranspose2d(16, 8, 3, 2, 1, 1)

        self.conv4 = nn.Conv2d(8, 1, 3, 1, 1)

    def forward(self, x):
        # x: [B,G,D,H,W]
        # out: [B,D,H,W]
        
        B, G, D, H, W = x.size()
        # change into appropriate form [B * D, H, W, G]
        s = x.permute(0, 2, 1, 3, 4).reshape(-1, G, H, W)

        c_0 = F.relu(self.conv1(s))
        c_1 = F.relu(self.conv2(c_0))
        c_2 = F.relu(self.conv3(c_1))
        c_3 = self.conv_trans1(c_2)
        c_4 = self.conv_trans2(c_3 + c_1)

        # [B * D, H, W, 1]
        s_bar = self.conv4(c_4 + c_0)
        s_bar = s_bar.view(B, D, H, W)

        return s_bar


def warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, D]
    # out: [B, C, D, H, W]
    B,C,H,W = src_fea.size()
    D = depth_values.size(1)
    # compute the warped positions with depth values
    with torch.no_grad():
        # relative transformation from reference to source view
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]
        y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, W, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(H * W), x.view(H * W)
        
        # turn into homogenous coordinates [3, H * W]
        grid = torch.stack((x, y, torch.ones_like(x)), dim=0)
        # apply rotation [B, 3, H * W]
        rot_grid = torch.matmul(rot, grid.unsqueeze(0).repeat(B, 1, 1))
        # unproj with depth values [B, 3, D, H * W] * [B, 1, D, 1] = [B, 3, D, H * W]
        depth_grid = rot_grid.unsqueeze(2).repeat(1, 1, D, 1) * depth_values.view(B, 1, D, -1)
        # translation [B, 3, D, H * W] + [B, 3, 1, 1] = [B, 3, D, H * W]
        trans_grid = depth_grid + trans.view(B, 3, 1, 1)
        # unhomo [B, 2, D, H * W] / [B, 1, D, H * W] = [B, 2, D, H * W]
        grid = trans_grid[:, :2, :, :] / trans_grid[:, -1:, :, :]
        # normalize into [-1, 1], [B, D, H * W]
        norm_x = grid[:, 0, :, :] / ((W - 1) / 2) - 1
        norm_y = grid[:, 1, :, :] / ((H - 1) / 2) - 1
        # [B, D, H * W, 2]
        grid = torch.stack((norm_x, norm_y), dim=-1)

    # since we're using 2d grid sample, we need to make change the shape of grid to [B, H', W, 2]
    # and then change back to proper shape
    warped_src_fea = F.grid_sample(
        src_fea, grid.view(B, H * D, W, 2), 
        mode='bilinear', 
        padding_mode='zeros',
        align_corners=False
    ).view(B, C, D, H, W)

    return warped_src_fea

def group_wise_correlation(ref_fea, warped_src_fea, G):
    # ref_fea: [B,C,H,W]
    # warped_src_fea: [B,C,D,H,W]
    # out: [B,G,D,H,W]
    B, C, D, H, W = warped_src_fea.size()

    # ref_fea [B, C, D, H, W] -> [B, D, H, W, C]
    # warped_src_fea [B, C, D, H, W] -> [B, D, H, W, C]
    grouped_ref_fea = ref_fea.unsqueeze(2).repeat(1, 1, D, 1, 1).permute(0, 2, 3, 4, 1).view(B, D, H, W, G, 1, -1)
    grouped_warped_src_fea = warped_src_fea.permute(0, 2, 3, 4, 1).view(B, D, H, W, G, -1, 1)
    
    # [B, D, H, W, G]
    correlation = torch.matmul(grouped_ref_fea, grouped_warped_src_fea).view(B, D, H, W, G).permute(0, 4, 1, 2, 3)

    return correlation


def depth_regression(p, depth_values):
    # p: probability volume [B, D, H, W]
    # depth_values: discrete depth values [B, D]
    # output [B, 1, H, W]

    B, D, H, W = p.size()

    p = p.permute(0, 2, 3, 1).view(B, H, W, 1, D)
    depth_values = depth_values.view(B, 1, 1, D, 1)

    # [B, H, W, 1, D] * [B, 1, 1, D, 1] = [B, H, W]
    expected_depth = torch.matmul(p, depth_values).view(B, H, W)

    return expected_depth

def mvs_loss(depth_est, depth_gt, mask):
    masked_depth_est = depth_est * mask
    return F.l1_loss(masked_depth_est, depth_gt)

