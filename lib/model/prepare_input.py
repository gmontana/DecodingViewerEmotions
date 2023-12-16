
import torch
from torch import nn
from torch.nn.init import normal_, constant_

import torchvision
import torchvision.transforms as T

from lib.model.motion import MDFM

def vis_MDM2(X,  save_folder, id):

    [b, m, seg, c, h, w] = X.size()


    for jb in range(b):
        for jm in range(m):
            for jn in range(seg):

                file_ID = f"{id}_b{jb}_m{jm}_n{jn}"
                T.ToPILImage()(X[jb,jm,jn]).save(f'{save_folder}/{file_ID}_MDM.png', mode='png')

def vis_MDM(X, X_MOT, save_folder, id):

    [b, m, n, t, c, h, w] = X.size()
    [b, m, n, c, h, w] = X_MOT.size()

    for jb in range(b):
        for jm in range(m):
            for jn in range(n):

                file_ID = f"{id}_b{jb}_m{jm}_n{jn}"
                T.ToPILImage()(X_MOT[jb,jm,jn]).save(f'{save_folder}/{file_ID}_MDM.png', mode='png')
                T.ToPILImage()(X[jb, jm, jn, t//2]).save(f'{save_folder}/{file_ID}_X.png', mode='png')

                T.ToPILImage()(X_MOT[jb,jm,jn, 0:1]).save(f'{save_folder}/{file_ID}_MDM_0.png', mode='png')
                T.ToPILImage()(X_MOT[jb,jm,jn, 1:2]).save(f'{save_folder}/{file_ID}_MDM_1.png', mode='png')
                T.ToPILImage()(X_MOT[jb,jm,jn, 2:3]).save(f'{save_folder}/{file_ID}_MDM_2.png', mode='png')

class X_input(torch.nn.Module):

    def __init__(self, param_TSM ):
        super(X_input, self).__init__()

        self.input_mode = param_TSM["main"]["input_mode"]
        self.n_video_segments = param_TSM["video_segments"] #[8, 8, 1]
        self.n_audio_segments = param_TSM["audio_segments"]  # [8, 8, 1]
        self.n_motion_segments = 0

        if param_TSM["motion"]:
            self.MDF = MDFM(param_TSM["motion_param"])
            self.n_motion_segments = self.n_video_segments

        print("self.n_motion_segments", self.n_motion_segments)

    def forward(self, x_video, x_audio):
        #print("X_input:", x_video.size(),  x_audio.size())
        #exit()

        with torch.no_grad():

            if self.n_motion_segments > 0:
                xMOT = self.MDF(x_video)
                #print("x_video", x_video.size())
                #print("xMOT", xMOT.size())
                #vis_MDM(x_video, xMOT, "logs/vis_MDM", "xMOT")
                #exit()

                if self.n_video_segments > 0:
                    k_frames = x_video.size()[3]
                    if k_frames > 1:
                        xRGB = x_video[:, :, :, k_frames // 2]
                    else:
                        xRGB = x_video
                    #print("xRGB", xRGB.size())

                    if self.input_mode  == 1:
                        x = torch.cat((xRGB, xMOT), dim=2)
                        #vis_MDM2(x, "logs/vis_MDM", "xMODE1")
                        #exit()
                    elif self.input_mode == 2:
                        x = torch.stack([torch.stack([xRGB[:, :, i], xMOT[:, :, i]]) for i in range(self.n_video_segments)])
                        x = x.view((-1,) + (x.size()[-5:])).permute((1, 2, 0, 3, 4, 5))
                        #vis_MDM2(x, "logs/vis_MDM", "xMODE2")
                        #exit()
                    else:
                        sys.exit(f'VCM_input: incorrect mode: {self.input_mode}')
                else:
                    x = xMOT

            elif self.n_video_segments  > 0:
                x = torch.squeeze(x_video, -4)

            if self.n_audio_segments > 0:
                if self.n_video_segments > 0:
                    x = torch.cat((x, x_audio), dim=2)
                else:
                    x = x_audio

        #print("x", x.size())
        #exit()
        return x




