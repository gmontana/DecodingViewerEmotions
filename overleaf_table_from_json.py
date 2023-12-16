import csv
import re
import argparse
import json

import collections
from collections import Counter


from mvlib.mvideo_lib  import Video, Emotions


def table_full_length_video(acc, eid, ename, valtest):
    backslash_char = bc = "\\"
    figure_char_A = fA = "{"
    figure_char_B = fB = "}"

    table_latex = f'{bc}begin{fA}table{fB}[t]\n'
    table_latex += f'{bc}caption{fA}Accuracy for localization of {ename} jumps in the full length video ads ({valtest} set) {fB}\n'
    table_latex += f'{bc}begin{fA}center{fB}\n'
    table_latex += f'{bc}begin{fA}small{fB}\n'
    table_latex += f'{bc}begin{fA}tabular{fB}{fA}lcccc{fB}\n'
    table_latex += f'{bc}toprule\n'

    columns = ["Score threshold", "Accuracy", "True positive", "Positive", "Predicted"]

    for col_name in columns:
        table_latex += f'{bc}makecell{fA}{col_name}{fB} & '
    table_latex += f'{bc}{bc}\n'
    table_latex += f'{bc}midrule\n'

    for t in acc[eid]:
        # print(t, acc[eid][t])
        score = f'{t}'
        acc_b = f'{round(acc[eid][t]["balanced"] * 100, 1)}'
        acc_u = f'{round(acc[eid][t]["unbalanced"] * 100, 1)}'
        TP = f'{acc[eid][t]["pred_true"]}'
        T = f'{acc[eid][t]["true"]}'
        P = f'{acc[eid][t]["pred"]}'

        table_latex += f'{score} & {acc_b}({acc_u}) & {TP} & {T} & {P} {bc}{bc}\n'

        if int(t) > 10:  break

    table_latex += f'{bc}bottomrule\n'
    table_latex += f'{bc}end{fA}tabular{fB}\n'
    table_latex += f'{bc}end{fA}small{fB}\n'
    table_latex += f'{bc}end{fA}center{fB}\n'
    table_latex += f'{bc}end{fA}table{fB}\n'

    return table_latex






def table_ROC_full_length_video(acc_val,acc_test, E, mode):
    #mode = ["f16", "with BackGround", "RGB_audio", "imagenet"]
    backslash_char = bc = "\\"
    figure_char_A = fA = "{"
    figure_char_B = fB = "}"

    table_latex = f'{bc}begin{fA}table{fB}[t]\n'
    table_latex += f'{bc}caption{fA} Area Under the Curve (AUC) for localization of emotion jumps in the full length video ads ({mode[1]},{mode[3]}) {fB}\n'
    table_latex += f'{bc}label{fA}table:AUC_{mode[1]}_{mode[3]}{fB}\n'
    table_latex += f'{bc}begin{fA}center{fB}\n'
    table_latex += f'{bc}begin{fA}small{fB}\n'
    table_latex += f'{bc}begin{fA}tabular{fB}{fA}lcccc{fB}\n'
    table_latex += f'{bc}toprule\n'

    columns = ["Emotion", "AUC \\ (validation set) " , "AUC \\ (test set) "]

    for col_name in columns:
        table_latex += f'{bc}makecell{fA}{col_name}{fB} & '
    table_latex += f'{bc}{bc}\n'
    table_latex += f'{bc}midrule\n'


    print("acc_val", acc_val)
    #exit()

    for eid in acc_val:
        ename = E.mapE[int(eid)]
        #print(eid, ename, acc_val[eid], acc_test[eid])
        #exit()

        val = f'{round(acc_val[eid], 2)}'
        test = f'{round(acc_test[eid], 2)}'


        table_latex += f'{ename} & {val} & {test} {bc}{bc}\n'



    table_latex += f'{bc}bottomrule\n'
    table_latex += f'{bc}end{fA}tabular{fB}\n'
    table_latex += f'{bc}end{fA}small{fB}\n'
    table_latex += f'{bc}end{fA}center{fB}\n'
    table_latex += f'{bc}end{fA}table{fB}\n'

    return table_latex



def table_RES_by_emotion(accE_val, accE_test, mode, E, best_modality):
    backslash_char = bc = "\\"
    figure_char_A = fA = "{"
    figure_char_B = fB = "}"

    table_latex = f'{bc}begin{fA}table{fB}[t]\n'
    table_latex += f'{bc}caption{fA} Emotional jumps: classification results split by emotion ({mode}). For the model: {best_modality[0]}, frames: {best_modality[1]} , pretrained: {best_modality[2]} {fB}\n'
    table_latex += f'{bc}label{fA}table:Emotional_jumps_classification_results_byemotion_{mode}{fB}\n'
    table_latex += f'{bc}begin{fA}center{fB}\n'
    table_latex += f'{bc}begin{fA}small{fB}\n'
    table_latex += f'{bc}begin{fA}tabular{fB}{fA}lcccc{fB}\n'
    table_latex += f'{bc}toprule\n'

    columns = ["Emotion",  f"Accuracy{bc}{bc}(validation)", f"Accuracy{bc}{bc}(test)"]
    for col_name in columns:
        table_latex += f'{bc}makecell{fA}{col_name}{fB} & '
    table_latex += f'{bc}{bc}\n'
    table_latex += f'{bc}midrule\n'

    for ename in accE_val:
        #ename = E.mapE[int(eid)]
        val = f'{round(accE_val[ename]*100, 1)}'
        test = f'{round(accE_test[ename]*100, 1)}'

        table_latex += f'{ename} & {val} & {test} {bc}{bc}\n'
    table_latex += f'{bc}bottomrule\n'
    table_latex += f'{bc}end{fA}tabular{fB}\n'
    table_latex += f'{bc}end{fA}small{fB}\n'
    table_latex += f'{bc}end{fA}center{fB}\n'
    table_latex += f'{bc}end{fA}table{fB}\n'
    return table_latex


def table_RES(acc, mode, E):
    backslash_char = bc = "\\"
    figure_char_A = fA = "{"
    figure_char_B = fB = "}"

    table_latex = f'{bc}begin{fA}table{fB}[t]\n'
    table_latex += f'{bc}caption{fA} Emotional jumps: classification results ({mode}) {fB}\n'
    table_latex += f'{bc}label{fA}table:Emotional_jumps_classification_results_{mode}{fB}\n'
    table_latex += f'{bc}begin{fA}center{fB}\n'
    table_latex += f'{bc}begin{fA}small{fB}\n'
    table_latex += f'{bc}begin{fA}tabular{fB}{fA}lcccc{fB}\n'
    table_latex += f'{bc}toprule\n'


    columns = ["Modality", "Frames" , "PreTrained", f"Accuracy{bc}{bc}(validation)", f"Accuracy{bc}{bc}(test)"]

    for col_name in columns:
        table_latex += f'{bc}makecell{fA}{col_name}{fB} & '
    table_latex += f'{bc}{bc}\n'
    table_latex += f'{bc}midrule\n'



    for i in acc:


        #print(i, acc[i]['validation'][3]['best_score']['balanced'])

        modality = f'{acc[i]["validation"][0]}'.replace("_", "+")
        #modality = modality
        frames = f'{acc[i]["validation"][1]}'
        PreTrained = f'{acc[i]["validation"][2]}'
        acc_val = float(acc[i]['validation'][3]['best_score']['balanced'])
        acc_test = float(acc[i]['test'][3]['best_score']['balanced'])

        table_latex += f'{modality} & {frames} & {PreTrained}  & {round(acc_val*100,1)} & {round(acc_test*100,1)} {bc}{bc}\n'

        if int(i) == 0:
            best_score = acc_test
            accE_val = acc[i]['validation'][3]['accE']
            accE_test = acc[i]['test'][3]['accE']
            best_modality = [modality, frames, PreTrained]


        if best_score < acc_test:
            best_score = acc_test
            accE_val = acc[i]['validation'][3]['accE']
            accE_test = acc[i]['test'][3]['accE']
            best_modality = [modality, frames, PreTrained]


    table_latex += f'{bc}bottomrule\n'
    table_latex += f'{bc}end{fA}tabular{fB}\n'
    table_latex += f'{bc}end{fA}small{fB}\n'
    table_latex += f'{bc}end{fA}center{fB}\n'
    table_latex += f'{bc}end{fA}table{fB}\n'

    print("best_score", best_score , accE_val)

    table_latex_by_emotion = table_RES_by_emotion(accE_val, accE_test, mode, E , best_modality)

    return table_latex, table_latex_by_emotion

def main():

    file_json = "paper_data/stat_clip_video.json"
    with open(file_json) as f:
        table_json = json.load(f)

    clips_tags = ["training","validation","test" ]
    clips_tags_all = ["training_all", "validation_all", "test_all"]
    video_tags_all = ["training_all", "validation_all", "test_all"]
    print(table_json["0.1"]['training'])
    str_table_start = '\\begin\n{table}[h!]\n\\begin{center}\n\\begin\n{tabular}\n{ccccc}\n\\hline\n'
    str_table_end = '\\end{tabular}\n\\end{center}\n\\caption{}\n\\label{}\n\\end{table}\n\n '

    file_table = f"paper_data/stat_clip_video.table"
    ft = open(file_table, "w")
    for porog in table_json:
        print("porog", porog)
        ft.write(f"table percentile: {porog}\n\n\n")
        statE = Counter()
        statV = Counter()
        totalC = 0
        for tag in clips_tags:
            for emotion in table_json[porog][tag]:
                statE[emotion] += table_json[porog][tag][emotion]
        for tag in clips_tags_all:
            totalC += table_json[porog][tag]

        for emotion in table_json[porog]["videos"]:
            statV[emotion] = table_json[porog]["videos"][emotion]
        totalV = table_json[porog]["videos_all"]
        print(statE)




        tHearder = f'Emotion & No of clips & No of videos \\\\\n'

        ft.write(str_table_start)
        ft.write(tHearder)
        ft.write('\\hline\n')
        for emotion in sorted(statE):
            ft.write(f'{emotion} & {statE[emotion]} & {statV[emotion]}  \\\\\n')
        ft.write('\\hline\n')
        ft.write(f'Total & {totalC} & {totalV}  \\\\\n')

        ft.write(str_table_end)

    ft.close()
        #exit()




if __name__ == '__main__':
    main()


             
        



