import torch.nn.functional as F
import torch.nn as nn
import torch
import sys
import math
from common.utils import poses_motion
from common.utils import EMG_diff



class c1D(nn.Module):
    # input (B,C,D) //batch,channels,dims
    # output = (B,C,filters)
    def __init__(self, input_channels, input_dims, filters, kernel):
        super(c1D, self).__init__()
        self.cut_last_element = (kernel % 2 == 0)
        self.padding = math.ceil((kernel - 1)/2)
        self.conv1 = nn.Conv1d(input_dims, filters,
                               kernel, bias=False, padding=self.padding)
        self.bn = nn.BatchNorm1d(num_features=input_channels)

    def forward(self, x):
        # x (B,D,C)
        x = x.permute(0, 2, 1)
        # output (B,filters,C)
        if(self.cut_last_element):
            output = self.conv1(x)[:, :, :-1]
        else:
            output = self.conv1(x)
        # output = (B,C,filters)
        output = output.permute(0, 2, 1)
        output = self.bn(output)
        output = F.leaky_relu(output, 0.2, True)
        return output



class block(nn.Module):
    def __init__(self, input_channels, input_dims, filters, kernel):
        super(block, self).__init__()
        self.c1D1 = c1D(input_channels, input_dims, filters, kernel)
        self.c1D2 = c1D(input_channels, filters, filters, kernel)

    def forward(self, x):
        output = self.c1D1(x)
        output = self.c1D2(output)
        return output



class d1D(nn.Module):
    def __init__(self, input_dims, filters):
        super(d1D, self).__init__()
        self.linear = nn.Linear(input_dims, filters)
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



class M2HGR(nn.Module):
    def __init__(self, frame_l, joint_n, joint_d, feat_d, feat_emg, filters, class_num):
        super(M2HGR, self).__init__()

        # EMG part
        ## part1：original EMG data
        self.emg_conv1_2_3 = nn.Sequential(
            c1D(frame_l, feat_emg, 2 * filters, 1),
            spatialDropout1D(0.1),
            c1D(frame_l, 2 * filters, filters, 3),
            spatialDropout1D(0.1),
            c1D(frame_l, filters, filters, 1)
        )
        self.emg_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            spatialDropout1D(0.1)
        )
        ## part2：temporal difference，stride=1
        ### same as part1, just use the emg_conv1_2_3 and emg_pool


        # JCD part
        self.jcd_conv1 = nn.Sequential(
            c1D(frame_l, feat_d, 2 * filters, 1),
            spatialDropout1D(0.1)
        )
        self.jcd_conv2 = nn.Sequential(
            c1D(frame_l, 2 * filters, filters, 3),
            spatialDropout1D(0.1)
        )
        self.jcd_conv3 = c1D(frame_l, filters, filters, 1)
        self.jcd_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            spatialDropout1D(0.1)
        )



        # two-scale motion: diff_slow part and fast part
        self.slow_conv1 = nn.Sequential(
            c1D(frame_l, joint_n * joint_d, 2 * filters, 1),
            spatialDropout1D(0.1)
        )
        self.slow_conv2 = nn.Sequential(
            c1D(frame_l, 2 * filters, filters, 3),
            spatialDropout1D(0.1)
        )
        self.slow_conv3 = c1D(frame_l, filters, filters, 1)
        self.slow_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            spatialDropout1D(0.1)
        )
        self.fast_conv1 = nn.Sequential(
            c1D(frame_l//2, joint_n * joint_d, 2 * filters, 1), spatialDropout1D(0.1))
        self.fast_conv2 = nn.Sequential(
            c1D(frame_l//2, 2 * filters, filters, 3), spatialDropout1D(0.1))
        self.fast_conv3 = nn.Sequential(
            c1D(frame_l//2, filters, filters, 1), spatialDropout1D(0.1))


        # after cat: fusion
        # the SE strcture will need these member variables.
        self.filters = filters
        self.r = 16
        self.Global_Avg_Pooling = nn.AdaptiveAvgPool2d((1,1))

        self.linear_SE_1 = nn.Linear(int(5 * self.filters), int(5 * self.filters / self.r))
        self.linear_SE_2 = nn.Linear(int(5 * self.filters / self.r), int(5 * self.filters))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # after fusion
        self.block1 = block(frame_l//2, 5 * filters, 2 * filters, 3)
        self.block_pool1 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2), spatialDropout1D(0.1))

        self.block2 = block(frame_l//4, 2 * filters, 4 * filters, 3)
        self.block_pool2 = nn.Sequential(nn.MaxPool1d(
            kernel_size=2), spatialDropout1D(0.1))

        self.block3 = nn.Sequential(
            block(frame_l//8, 4 * filters, 8 * filters, 3), spatialDropout1D(0.1))

        self.linear1 = nn.Sequential(
            d1D(8 * filters, 128),
            nn.Dropout(0.5)
        )
        self.linear2 = nn.Sequential(
            d1D(128, 128),
            nn.Dropout(0.5)
        )

        self.linear3 = nn.Linear(128, class_num)



    def forward(self, M, P, N):
        x = self.jcd_conv1(M)
        x = self.jcd_conv2(x)
        x = self.jcd_conv3(x)
        x = x.permute(0, 2, 1)
        # pool will downsample the D dim of (B,C,D)
        # but we want to downsample the C channels
        # 1x1 conv may be a better choice
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
        # x,x_d_fast,x_d_slow shape: (B,framel//2,filters)

        # EMG part: "N" as EMG input
        ## part1: original EMG data
        x_emg_1 = self.emg_conv1_2_3(N)
        x_emg_1 = x_emg_1.permute(0, 2, 1)
        x_emg_1 = self.emg_pool(x_emg_1)
        x_emg_1 = x_emg_1.permute(0, 2, 1)
        ## part2：temporal difference，stride=1
        x_emg_2 = EMG_diff(N)
        x_emg_2 = self.emg_conv1_2_3(x_emg_2)
        x_emg_2 = x_emg_2.permute(0, 2, 1)
        x_emg_2 = self.emg_pool(x_emg_2)
        x_emg_2 = x_emg_2.permute(0, 2, 1)


        # SE part
        x = torch.cat((x, x_d_slow, x_d_fast, x_emg_1, x_emg_2), dim=2)
        x = x.permute(2, 0, 1)                                            
        h = self.Global_Avg_Pooling(x)
        x = x.permute(1, 2, 0)
        h = h.permute(1, 2, 0)
        h = self.linear_SE_1(h)
        h = self.relu(h)
        h = self.linear_SE_2(h)
        h = self.sigmoid(h)
        h = h * x

        x = x + h

        x = self.block1(x)
        x = x.permute(0, 2, 1)
        x = self.block_pool1(x)
        x = x.permute(0, 2, 1)

        x = self.block2(x)
        x = x.permute(0, 2, 1)
        x = self.block_pool2(x)
        x = x.permute(0, 2, 1)

        x = self.block3(x)
        # max pool over (B,C,D) C channels
        x = torch.max(x, dim=1).values

        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x
