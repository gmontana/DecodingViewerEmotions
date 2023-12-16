import collections
from collections import Counter


import matplotlib.pyplot as plt
import math
import os
import numpy as np


def plot_clicks_per_video_per_user_per_second(APE_meanstd,  E, file, xlabel=None, ylabel=None, fontsize=18, plotdir=None):

    if plotdir is not None:
        plotdir = plotdir
        os.makedirs(plotdir, exist_ok=True)
    else:
        plotdir = "plots/"
        os.makedirs(plotdir, exist_ok=True)

    labels = []
    CTEs = []
    error = []
    x = []

    for EID in range(1, 9):
        [avg,sd] = APE_meanstd[EID]
        CTEs.append(avg)
        error.append(sd)
        x.append(E[EID])

    plt.errorbar(x, CTEs, yerr=error, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)

    title = f"Clicks per video (adjusted)"
    plt.title(title, fontsize=fontsize)


    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig(f'{plotdir}/{file}.eps' , format = 'eps', dpi=900)
    plt.savefig(f'{plotdir}/{file}.png')
    plt.close()


def plot_clicks_per_video_per_user_per_second_All_vs_Neutral(APE_meanstd,  E, file, xlabel=None, ylabel=None, fontsize=18, plotdir=None):

    if plotdir is not None:
        plotdir = plotdir
        os.makedirs(plotdir, exist_ok=True)
    else:
        plotdir = "plots/"
        os.makedirs(plotdir, exist_ok=True)

    labels = []
    CTEs = []
    error = []
    x = []

    for EID in range(1, 10):
        if EID not in [6,9]:continue
        [avg, sd] = APE_meanstd[EID]
        CTEs.append(avg)
        error.append(sd)
        x.append(E[EID])


    plt.figure(figsize=(2.5, 4.5))

    plt.errorbar(x, CTEs, yerr=error, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=3, markersize=3)
    #plt.axhline(y=0.0, color='k', linestyle='-', lw=1)

    plt.xlim([-0.5, 1.5])
    title = f""
    plt.title(title, fontsize=fontsize)




    #if xlabel is not None: ax.xlabel(xlabel)
    if ylabel is not None: plt.ylabel(ylabel)

    # Save the figure and show

    plt.tight_layout()
    plt.savefig(f'{plotdir}/{file}.eps' , format = 'eps', dpi=900)
    plt.savefig(f'{plotdir}/{file}.png')
    plt.close()


def plot_video_clics(valueList,  N, title, file, xfactor = 1,xmax = None , xlabel=None, ylabel=None, fontsize=18, plotdir=None):

    if plotdir is not None:
        plotdir = plotdir
        os.makedirs(plotdir, exist_ok=True)
    else:
        plotdir = "plots_video_oct22/"
        os.makedirs(plotdir, exist_ok=True)

    round_k = int( math.log10( xfactor ))
    #xmax = 0
    #print("zaeblo")
    list_plot_y, list_plot_x = [], []
    for k in sorted(valueList):
        #print("zaeblo", k, valueList[k], N)

        list_plot_y.append(100*int(valueList[k])/N)
        list_plot_x.append(round(k/xfactor, round_k))
        #if k > xmax: xmax = k


    title = f"{title}"
    plt.title(title, fontsize=fontsize)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    barlist = plt.bar(list_plot_x, list_plot_y)

    plt.xlim([0, xmax])
    #plt.ylim((0, 1000))
    plt.savefig(f'{plotdir}/{file}.eps' , format = 'eps', dpi=900)
    #plt.savefig(f'{plotdir}/{file}.png')
    plt.close()


def plot_clicks(clicks, eid, E, file, xmax = None , xlabel="Number of Clicks", ylabel="Percent rate", fontsize=12, plotdir=None):
    #clicks_eid[i_profile_clicks_eid[eid]] += 1
    print("plot_clicks:", plotdir)
    if plotdir is not None:
        plotdir = plotdir
        os.makedirs(plotdir, exist_ok=True)
    else:
        plotdir = "plots/"
        os.makedirs(plotdir, exist_ok=True)


    x, y = [], []
    x.append(-1)
    y.append(0)
    for n in sorted(clicks):
        x.append(n)
        y.append(clicks[n])
        if n > 4:
            break


    plt.bar(x, y, width=0.5)

    title = f"{E.mapE[eid]}"
    plt.title(title, fontsize=fontsize)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    #plt.xlim([0, 5])
    #plt.ylim([0, 100])
    x_labels = []
    for v in x:
        if v < 0:
            x_labels.append(f"")
        else:
            x_labels.append(f"{v}")
    plt.xticks(x, x_labels)

    plt.title(title, fontsize=fontsize)
    plt.savefig(f'{plotdir}/{file}.eps', format = 'eps', dpi=600)
    #plt.savefig(f'{plotdir}/{file}.png')
    plt.close()


def plot_aggregatedprofiles_meansd(APE_mean, APE_std, eid, E, file, xlabel="time", ylabel="clicks", fontsize=12, plotdir=None):

    if plotdir is not None:
        plotdir = plotdir
        os.makedirs(plotdir, exist_ok=True)
    else:
        plotdir = "plots/"
        os.makedirs(plotdir, exist_ok=True)

    labels = []
    CTEs = []
    error = []
    x = []

    for i, v in enumerate(APE_mean):
        CTEs.append(APE_mean[i])
        error.append(APE_std[i])
        x.append(i)
    plt.ylim([0, 1.50])
    plt.errorbar(x, CTEs, yerr=error, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)

    title = f"{E.mapE[eid]}"
    plt.title(title, fontsize=fontsize)

    ylabel = "Clicks per user"
    xlabel = "Time (Seconds)"

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig(f'{plotdir}/{file}.eps' , format = 'eps', dpi=300)
    plt.savefig(f'{plotdir}/{file}.png')
    plt.close()


def plot_CER(valueList, top, eid, E, file,  xlabel=None, ylabel=None, fontsize=12, plotdir=None):

    if plotdir is not None:
        plotdir = plotdir
        os.makedirs(plotdir, exist_ok=True)
    else:
        plotdir = "plots/"
        os.makedirs(plotdir, exist_ok=True)


    list_plot_y, list_plot_x = [], []
    for k in sorted(valueList):
        list_plot_y.append(valueList[k])
        list_plot_x.append(k)
        if valueList[k] < 0.01:
            xmax = k
            break

    title = f"{E.mapE[eid]}"
    plt.title(title, fontsize=fontsize)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    barlist = plt.bar(list_plot_x, list_plot_y)



    for rect in barlist:
        #print("red", rect.xy[0] , V_TOP)
        if rect.xy[0] > top:
            rect.set_color('r')
        #elif rect.xy[0] < V_LOW*100:
        #    rect.set_color('blue')
        else:
            rect.set_color('black')


    plt.xlim([0, xmax])
    plt.ylim((0, 10))
    plt.savefig(f'{plotdir}/{file}.eps' , format = 'eps', dpi=300)
    plt.savefig(f'{plotdir}/{file}.png')
    plt.close()


def plot_S2N(S2N, step, eid, E, file,  xlabel=None, ylabel=None, fontsize=12, plotdir=None):
    print("plot_S2N", S2N, step)
    if plotdir is not None:
        plotdir = plotdir
        os.makedirs(plotdir, exist_ok=True)
    else:
        plotdir = "plots/"
        os.makedirs(plotdir, exist_ok=True)


    list_plot_y, list_plot_x = [], []
    #for k in sorted(S2N):
    #for k in sorted(S2N):
    for k in S2N:
        Y = S2N[k]

        if float(k)*100 < 10:
            list_plot_y.append(Y)
            list_plot_x.append(round(float(k) * 100,1))
            print("plot_S2N", float(k) * 100, S2N[k], step)



        #if valueList[k] < 0.01:

        #    break

    title = f"{E.mapE[eid]}"
    plt.title(title, fontsize=fontsize)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    plt.bar(list_plot_x, list_plot_y , width=0.1)
    x_labels = []
    for v in list_plot_x:
        x_labels.append(f"{v}-{v-0.5}")
    plt.xticks(list_plot_x, x_labels, rotation='vertical')

    #plt.xlim([0, len(x_labels)+1])
    plt.ylim((0, 7))
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(f'{plotdir}/{file}.eps', format = 'eps', dpi=300)
    plt.savefig(f'{plotdir}/{file}.png')
    plt.close()


def plot_ROC(fpr,tpr,roc_auc, eid, E, id, file, plotdir=None):


    if plotdir is not None:
        plotdir = plotdir
        os.makedirs(plotdir, exist_ok=True)
    else:
        plotdir = "plots/ROC"
        os.makedirs(plotdir, exist_ok=True)

    plt.figure()

    """
    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i],tpr[i],color=color,lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )
    """

    title = f"{E.mapE[eid]}"
    plt.title(title, fontsize=12)
    lw = 2
    label = f"ROC curve model: ({id}, area = {round(roc_auc,2)})"
    plt.plot(fpr,tpr, color="darkorange", lw=lw , label = label)#label="ROC curve (area = {1:0.2f})".format(roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.legend(loc="lower right")
    plt.savefig(f'{plotdir}/{file}.eps', format='eps', dpi=300)
    plt.savefig(f'{plotdir}/{file}.png')

    #plt.show()


def plot_profile(APT_e, t_max, title, file, xlabel="time", ylabel="emotion level", fontsize=16, plotdir=None):

    if plotdir is not None:
        plotdir = plotdir
    else:
        plotdir = "plots/"
        os.makedirs(plotdir, exist_ok=True)

    list_plot_x, list_plot_y = [], []
    for ti in range(len(APT_e)):
        list_plot_y.append(APT_e[ti])
        list_plot_x.append(ti)

    plt.title(title, fontsize=fontsize)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    plt.plot(list_plot_x, list_plot_y, '-p', color='blue', markersize=2, linewidth=1,)
    plt.axvline(x=t_max-1, color='red')
    plt.axvline(x=t_max+5-1, color='red')
    plt.savefig(f'{plotdir}/{file}.eps', format='eps', dpi=900)
    
    plt.close()









             
        



