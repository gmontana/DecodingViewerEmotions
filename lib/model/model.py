
import torch
from torch import nn


class VCM(torch.nn.Module):

    def __init__(self, param_TSM, nn_X_input, nn_backbone):
        super(VCM, self).__init__()

        self.model_layers = {}

        self.nn_X_input = nn_X_input
        self.model_layers["nn_X_input"] = self.nn_X_input

        self.nn_backbone = nn_backbone
        self.model_layers["nn_backbone"] = self.nn_backbone

        self.num_class = param_TSM["num_class"]
        self.n_video_segments = param_TSM["video_segments"]  # 8
        self.n_audio_segments = param_TSM["audio_segments"]  # 1
        self.n_motion_segments = 0  # 0, or == self.n_video_segments
        if param_TSM["motion"]:
            self.n_motion_segments = self.n_video_segments

        # self.n_seg = self.n_video_segments + self.n_audio_segments + self.n_motion_segments
        # self.last_num = self.nn_backbone.features_dim_out * self.n_seg
        self.last_num = self.nn_backbone.features_dim_out

        print("last_fc", self.last_num, self.num_class)

        self.last_fc = nn.Linear(self.last_num, self.num_class)
        self.model_layers["last_fc"] = self.last_fc
        # self.last_activation = torch.nn.ReLU()

    def forward(self, x_video, x_audio):
        # print("VCM INPUT:", x_video.size())

        x = self.nn_X_input(x_video, x_audio)

        # print("VCM self.nn_X_input:", x.size())

        n_seg = x.size()[-4]
        n_sample = x.size()[-5]

        # print("VCM n_seg, n_sample :", n_seg, n_sample )

        x = self.nn_backbone(x)  # , xA

        # print("nn_backbone(x) :", x.size())
        # exit()

        """ average accross n_sample"""
        x = x.mean(dim=1)
        # print("average accross n_sample :", x.size())

        """ average accross n_seg"""
        x = x.mean(dim=1)
        # print("average accross n_seg :", x.size())
        # exit()

        x = x.view((-1, self.last_num))

        # print("before fc :", x.size())

        # x = self.last_activation(x)

        x = self.last_fc(x)
        # x = torch.add(x, 1)
        # x = x.pow(2)
        # print("x:", x.size())

        # print("after fc :", x.size() , x[0].sum(), x[1].sum() )
        # print("x:", x)
        # exit()

        # x = x.view((-1,n_sample, x.size()[-1]))

        # print("after fc :", x.size())

        # """ average accross n_sample"""
        # x = x.mean(dim=1)
        # exit()

        return x  # , xA
