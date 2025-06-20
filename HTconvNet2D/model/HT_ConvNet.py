from utils import poses_motion
import torch.nn.functional as F
import torch.nn as nn
import torch
import sys
import math

class AttentionFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionFusion, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 3)

    def forward(self, jcd_features, slow_features, fast_features):
        concatenated_features = torch.cat((jcd_features, slow_features, fast_features), dim=2)
        attention_weights = F.softmax(self.fc2(F.relu(self.fc1(concatenated_features))), dim=2)

        weighted_jcd = attention_weights[:,:,0].unsqueeze(2) * jcd_features
        weighted_slow = attention_weights[:,:,1].unsqueeze(2) * slow_features
        weighted_fast = attention_weights[:,:,2].unsqueeze(2) * fast_features

        fused_features = weighted_jcd + weighted_slow + weighted_fast
        return fused_features

class c1D(nn.Module):
    def __init__(self, input_channels, input_dims, filters, kernel, dilation=1):
        super(c1D, self).__init__()
        self.cut_last_element = (kernel % 2 == 0)
        self.padding = dilation * (kernel - 1) // 2
        self.conv1 = nn.Conv1d(input_dims, filters, kernel, bias=False, padding=self.padding, dilation=dilation)
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='leaky_relu')
        self.bn = nn.BatchNorm1d(num_features=input_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        if self.cut_last_element:
            output = self.conv1(x)[:, :, :-1]
        else:
            output = self.conv1(x)
        output = output.permute(0, 2, 1)
        output = self.bn(output)
        output = F.leaky_relu(output, 0.2, True)
        return output

class block(nn.Module):
    def __init__(self, input_channels, input_dims, filters, kernel, dilation=1):
        super(block, self).__init__()
        self.c1D1 = c1D(input_channels, input_dims, filters, kernel, dilation)
        self.c1D2 = c1D(input_channels, filters, filters, kernel, dilation)

    def forward(self, x):
        output = self.c1D1(x)
        output = self.c1D2(output)
        return output

class d1D(nn.Module):
    def __init__(self, input_dims, filters):
        super(d1D, self).__init__()
        self.linear = nn.Linear(input_dims, filters)
        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='leaky_relu')
        self.bn = nn.BatchNorm1d(num_features=filters)

    def forward(self, x):
        output = self.linear(x)
        output = self.bn(output)
        output = F.leaky_relu(output, 0.2)
        return output

class spatialDropout1D(nn.Module):
    def __init__(self, p):
        super(spatialDropout1D, self).__init__()
        self.dropout = nn.Dropout2d(p)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        return x

class HT_ConvNet(nn.Module):
    def __init__(self, frame_l, joint_n, joint_d, feat_d, filters, class_num):
        super(HT_ConvNet, self).__init__()
        self.jcd_conv1 = nn.Sequential(
            c1D(frame_l, feat_d, 2 * filters, 1, dilation=1),
            spatialDropout1D(0.1)
        )
        self.jcd_conv2 = nn.Sequential(
            c1D(frame_l, 2 * filters, filters, 3, dilation=2),
            spatialDropout1D(0.1)
        )
        self.jcd_conv3 = c1D(frame_l, filters, filters, 1, dilation=4)
        self.jcd_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            spatialDropout1D(0.1)
        )

        self.slow_conv1 = nn.Sequential(
            c1D(frame_l, joint_n * joint_d, 2 * filters, 1, dilation=1),
            spatialDropout1D(0.1)
        )
        self.slow_conv2 = nn.Sequential(
            c1D(frame_l, 2 * filters, filters, 3, dilation=2),
            spatialDropout1D(0.1)
        )
        self.slow_conv3 = c1D(frame_l, filters, filters, 1, dilation=4)
        self.slow_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            spatialDropout1D(0.1)
        )

        self.fast_conv1 = nn.Sequential(
            c1D(frame_l//2, joint_n * joint_d, 2 * filters, 1, dilation=1),
            spatialDropout1D(0.1)
        )
        self.fast_conv2 = nn.Sequential(
            c1D(frame_l//2, 2 * filters, filters, 3, dilation=2),
            spatialDropout1D(0.1)
        )
        self.fast_conv3 = c1D(frame_l//2, filters, filters, 1, dilation=4)

        self.attention_fusion = AttentionFusion(filters * 3, filters)

        self.block1 = block(frame_l//2, filters, 2 * filters, 3, dilation=1)
        self.block_pool1 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            spatialDropout1D(0.1)
        )

        self.block2 = block(frame_l//4, 2 * filters, 4 * filters, 3, dilation=1)
        self.block_pool2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            spatialDropout1D(0.1)
        )

        self.block3 = nn.Sequential(
            block(frame_l//8, 4 * filters, 8 * filters, 3, dilation=1),
            spatialDropout1D(0.1)
        )

        self.linear1 = nn.Sequential(
            d1D(8 * filters, 128),
            nn.Dropout(0.5)
        )
        self.linear2 = nn.Sequential(
            d1D(128, 128),
            nn.Dropout(0.5)
        )

        self.linear3 = nn.Linear(128, class_num)

    def forward(self, M, P=None):
        x = self.jcd_conv1(M)
        x = self.jcd_conv2(x)
        x = self.jcd_conv3(x)
        x = x.permute(0, 2, 1)
        x = self.jcd_pool(x)
        x = x.permute(0, 2, 1)

        diff_slow, diff_fast = poses_motion(P)
        x_d_slow = self.slow_conv1(diff_slow)
        x_d_slow = self.slow_conv2(x_d_slow)
        x_d_slow = self.slow_conv3(x_d_slow)
        x_d_slow = x_d_slow.permute(0, 2, 1)
        x_d_slow = self.slow_pool(x_d_slow)
        x_d_slow = x_d_slow.permute(0, 2, 1)

        x_d_fast = self.fast_conv1(diff_fast)
        x_d_fast = self.fast_conv2(x_d_fast)
        x_d_fast = self.fast_conv3(x_d_fast)

        x = self.attention_fusion(x, x_d_slow, x_d_fast)
        x = self.block1(x)
        x = x.permute(0, 2, 1)
        x = self.block_pool1(x)
        x = x.permute(0, 2, 1)

        x = self.block2(x)
        x = x.permute(0, 2, 1)
        x = self.block_pool2(x)
        x = x.permute(0, 2, 1)

        x = self.block3(x)
        x = torch.max(x, dim=1).values

        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x