import argparse
import json
import os

import statistics
import random
from collections import Counter

from mvlib.utils import save_pickle, load_pickle
from mvlib.mvideo_lib import VideoDB, StatVDB, Video , Emotions
#from mvlib.plots import plot_ROC

from lib.utils.utils import loadarg



def copy_frames(dir_videos, image_tmpl,BESTP_E , path_frame_folder):
    #f"{directory}/{image_tmpl.format(idx)}"
    os.mkdir(f"{path_frame_folder}")
    E = Emotions()
    for eid in range(1,9):
        emotion = E.mapE[eid]
        if os.path.exists(f"{path_frame_folder}/{emotion}"):
            os.rmdir(f"{path_frame_folder}/{emotion}")
            os.mkdir(f"{path_frame_folder}/{emotion}")
        else:
            os.mkdir(f"{path_frame_folder}/{emotion}")

        for [ID, t_max, v_max] in  BESTP_E[eid]:
            file_video = f"{dir_videos}/{ID}.mp4"

            cmd = f'scp {file_video} {path_frame_folder}/{emotion}/.'
            print(cmd)
            os.system(cmd)
            #fps = 10
            #cmd_frames = f"ffmpeg  -i {file_video}  -vf \"select='between(t,{t_max},{t_max+5})',fps={fps}\" -q:v 0 \"{path_frame_folder}/{E.mapE[eid]}/{ID}/{image_tmpl}.jpg\" "
            #print(cmd_frames)
            #exit()


def main():

    global args_model, args_data

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)    # data for prediction
    parser.add_argument("--file", type=str)    # data for prediction
    parser.add_argument("--fmode", type=str)  # data for prediction
    args_in = parser.parse_args()

    """plot Stat Figures"""
    if args_in.config:
        args = loadarg(args_in.config)

        if "fileVDB" in args["dataset"]:
            clip_length = args["emotion_jumps"]["clip_length"]
            jump = args["emotion_jumps"]["jump"]
            fileVDB = f'{args["dataset"]["fileVDB"]}'
            VDB = load_pickle(fileVDB)
            V_Durations = VDB.get_Durations()
            V_StarRate = VDB.get_StarRate()
            print("V_Durations", len(V_Durations))
            print("V_StarRate", len(V_StarRate))

            VDB.add_Emotions()
            VDB.get_dAPTV(clip_length)
            #self.max_Jump_Video[eid][VID]
            VDB.get_dAPTV_porogs(jump, type="top")
            #self.porogs[eid]

            # self.max_Jump_Video[eid][VID]
            E = Emotions()
            jmps = [3, 2, 1, 0.5, 0.1]
            for jV in jmps:
                VDB.get_video_with_jump(jV)
                fstr = ""
                for eid in VDB.video_with_jump:
                    fstr += f'"{E.mapE[eid]}": {VDB.video_with_jump[eid]}, '
                print(fstr)

                print("all:",len(VDB.VID_with_jump))
            exit()

            E = Emotions()
            line = {}
            Header = f"Rating & No of videos & "
            for eid in VDB.porogs:
                Header += f"{E.mapE[eid]} & "
                statSRP, statAll = Counter(), Counter()
                porog = VDB.porogs[eid]
                for VID in V_StarRate:
                    SR = V_StarRate[VID]
                    if VID in VDB.max_Jump_Video[eid]:
                        MaxV = VDB.max_Jump_Video[eid][VID]
                    else:
                        print("Netu", VID)
                        continue

                    for SRP in range(5,0,-1):
                        if SR >= SRP:
                            C = SRP
                            break

                    #print(VID, SR, MaxV, "C:", C)
                    if MaxV >= porog:
                        statSRP[C] += 1
                    statAll[C] += 1
                for C in sorted(statAll):

                    print(eid, E.mapE[eid], f"{C}+", statSRP[C], statAll[C], f"{round(100*(statSRP[C]/ statAll[C]),1)} %" )
                    #All[f"{C}+"][E.mapE[eid]] = statAll[C]

                    if eid == 1:
                        line[C] = f"{C}+ & "
                        line[C] += f"{statAll[C]} & "




                    line[C] += f"{round(100 * (statSRP[C] / statAll[C]), 1)} \\% &"



            print(f"{Header} \\\\")
            for C in sorted(statAll):
                print(f"{line[C]} \\\\")
                #exit()




            #exit()

            std_duration = [30, 15,60,20,10,120,20,40,5,45, 75]
            max = 120

            StatDuration = Counter()
            for item in V_Durations:
                print(item, V_Durations[item])
                T = V_Durations[item]
                for t in std_duration:

                    if abs(T -t) <=3:
                        T = t


                if T not in  std_duration:
                    T = 500
                #if abs(V_Durations[item] > max):
                    #V_Durations[item] = 150

                StatDuration[T] +=1

            print("StatDuration", StatDuration)
            for T in sorted(StatDuration):
                #print(T, StatDuration[T])
                print(f"{T} & {StatDuration[T]} \\\\")
            #exit()

            VDB.add_Emotions()
            VDB.get_dAPTV(clip_length)


            VDB.get_dAPTV_porogs(jump, type="top")




            SVDB = StatVDB(VDB)

            # plotdir="plots/CER/
            SVDB.CER_distribution(clip_length, jump)

            SVDB.clicks_figure7(V_Durations)

            #[ID, t_max, v_max]
            BESTP_E = SVDB.profiles_plot(V_Durations)



            exit()

            dir_videos = f'{args["dataset"]["data_dir"]}/{args["dataset"]["dir_videos"]}'
            image_tmpl = args["dataset"]["video_img_param"]["image_tmpl"]
            path_frame_folder = "tmp_examples_jumps"
            copy_frames(dir_videos, image_tmpl , BESTP_E, path_frame_folder)
            exit()

            SVDB.clicks_videos(V_Durations)

            # plotdir="plots/clicks"
            SVDB.clicks_stat()
            exit()

            # "plots/Signal2Noise/"
            SVDB.Signal2NoiseDistribution(clip_length)
            exit()

            # plotdir="plots/clicks"
            SVDB.clicks_stat()

            # plotdir="plots/aggregatedprofiles_meansd"
            SVDB.aggregated_profiles_meansd()

            # plotdir="plots/CER/
            SVDB.CER_distribution(clip_length, jump)



    if args_in.fmode:
        mode_str = args_in.fmode.replace("PPPPP", ' ')
        mode = mode_str.split("___")
        print("mode_str", mode)
        #exit()
    """plot ROC curves """
    if args_in.file:
        with open(args_in.file) as f:
            acc = json.load(f)

            #print(acc)
            #acc[eid][porog]
            #acc = {}
            #acc["unbalanced"] = accuracy_score(y_true, y_pred)
            #acc["balanced"] = balanced_accuracy_score(y_true, y_pred)
            #acc["pred_true"] = sum(y_TP)
            #acc["true"] = sum(y_true)
            #acc["pred"] = sum(y_pred)

            #TP_Rate = sum(y_TP) / P
            #FP_Rate = sum(y_FP) / N

            #acc["TP_Rate"] = TP_Rate
            #acc["FP_Rate"] = FP_Rate
            from sklearn.metrics import roc_curve, auc
            import numpy
            for eid in acc:
                TPR, FPR = [] , []
                for t in acc[eid]:
                    #print(t)
                    FPR.append(acc[eid][t]["FP_Rate"])
                    TPR.append(acc[eid][t]["TP_Rate"])

                FPR.reverse()
                TPR.reverse()
                FPR.append(1)
                TPR.append(1)

                print("FPR", FPR)


                FPR = numpy.array(FPR)
                TPR = numpy.array(TPR)
                #print("FPR", FPR)

                #print("TPR", TPR)

                roc_auc = auc(FPR, TPR)
                print(eid, roc_auc )
                eid = int(eid)
                E = Emotions()

                BG = "True"
                if mode[1] == 'no BackGround': BG = "False"
                id = f'{mode[3]}, {mode[4]}' #, modality: {mode[3].replace("_","+")}, BackGround: {BG}
                file = f"{E.mapE[eid]}_{mode[4]}_{mode[3]}_BG_{BG}"
                plot_ROC(FPR, TPR, roc_auc, eid, E, id, file, plotdir=None)
            #exit()






if __name__ == '__main__':
    main()








