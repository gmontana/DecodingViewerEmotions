import csv
import re
import numpy as np
import os


import collections
from collections import Counter

import statistics

from mvlib.utils  import save_pickle, load_pickle , save_dict, create_clean_DIR
from mvlib.video2frames  import parse_video_set

def parse_emstring(estr):
    #print("EmotionString", estr)
    estr_list = estr.split('|')
    ind_em_profile = []
    for it in estr_list:
        #print("it", it)
        if len(it) > 3:
            if "#" not in it:
                print("ERROR EmotionString", EmotionString)
                continue
            it_l = it.split('#')
            t = float(it_l[0])
            eid = int(it_l[1])
            #print("t,eid", t, int(t), eid)
            ind_em_profile.append([int(t),eid])
    return ind_em_profile

def get_value(s):
    s = s.replace('\'', '')
    return s




def get_dAPTV_EID(APTV, emotion_id, clip_length):
    # self.profile_anotation = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
    # APTV = load_pickle(path_Adcumen_APTV)
    # APTV[VID][eid][ti] = e_clicks / num_users

    dAPTV, stat = {}, Counter()
    max_Jump_Vid = {}
    for i, ID in enumerate(APTV):
        dAPTV[ID] = {}
        MAX_VID_dD =0
        for t in sorted(APTV[ID][emotion_id]):
            if t + clip_length in APTV[ID][emotion_id]:
                dD = APTV[ID][emotion_id][t + clip_length] - APTV[ID][emotion_id][t]
                if dD > MAX_VID_dD: MAX_VID_dD = dD
                dAPTV[ID][t] = APTV[ID][emotion_id][t + clip_length] - APTV[ID][emotion_id][t]
                stat[round(dAPTV[ID][t], 3)] += 1
        max_Jump_Vid[ID] = MAX_VID_dD
        #print(ID, MAX_VID_dD,  emotion_id)
    return dAPTV, stat, max_Jump_Vid

class Emotions:
    
    def __init__(self ):
        self.list = ["Anger" ,"Contempt" , "Disgust" , "Fear" ,"Happiness" , "Neutral" , "Sadness" , "Surprise"]
        self.mapE = {}
        self.mapE[-1] = "BackGround"
        for i, e in enumerate(self.list):
            self.mapE[i+1] = e
            print("Emotions", i, e)



def clicks_per_emotion(i_profile_V,Duration):

    clicks, clicks_peruser, clicks_peruser_persecond  = Counter(), Counter(), Counter()

    for i_profile in i_profile_V:
        clicks[0] += len(i_profile)
        for [t, eid] in i_profile:
            if t == 0:
                if eid == 6:
                    continue
            for EID in range(1, 9):
                if EID == eid:
                    clicks[EID] += 1

    for EID in range(0, 9):
        clicks_peruser[EID] = round(100* clicks[EID] / len(i_profile_V), 0)
        clicks_peruser_persecond[EID] = round( 100* clicks_peruser[EID] / Duration, 0)
        #print(EID, clicks[EID], clicks_peruser[EID], clicks_peruser_persecond[EID])
    #exit()

    return clicks, clicks_peruser, clicks_peruser_persecond


def get_video_profile_per_emotion(VID, i_profile_V,Duration ):
    Profile = {}
    for e in range(0, 9):
        Profile[e] = {}
        
        for ti in range(int(Duration)):
            Profile[e][ti] = 0

        
    for i_profile in i_profile_V:
        t_B = 0
        eid_B = 0
        for [t, eid] in sorted(i_profile,key=lambda x: x[0]):
            if t < t_B:
                print("ERROR", VID, t, eid, t_B, eid_B )
                exit()
            for ti in range(int(t_B), int(t)):
                Profile[eid_B][ti] +=1
            t_B = t
            eid_B = eid

        for ti in range(int(t_B), int(Duration)):
            Profile[eid_B][ti] += 1
        
        
    for e in range(0, 9):
        for ti in range(int(Duration)):
            Profile[e][ti] = Profile[e][ti] / len(i_profile_V)
            #print("i_profile_V:", e,ti, Profile[e][ti], len(i_profile_V))
    #exit()

    return Profile



class StatVDB:

    def __init__(self, VDB ):
        self.VDB = VDB
        self.E = Emotions()

    def profiles_plot(self, V_Durations):
        #self.IndividualProfiles[VID] = [t,eid]
        BESTP_E = {}
        for eid in  self.VDB.dAPTV:
            print("eid", eid)
            BESTP_E[eid] = []
            BESTP = []
            for ID in self.VDB.dAPTV[eid]:
                #
                if len(self.VDB.dAPTV[eid][ID]) > 0:
                    t_max = max(self.VDB.dAPTV[eid][ID], key=self.VDB.dAPTV[eid][ID].get)
                    v_max = self.VDB.dAPTV[eid][ID][t_max]
                    if v_max > 0:
                        BESTP.append([ID, t_max, v_max])
                        #print("eid ID", eid, ID, t_max, v_max)
            #print(BESTP[0:10])
            for i, [ID, t_max, v_max] in enumerate( sorted(BESTP, key=lambda x: x[2], reverse=True) ):
                print("BESTP_E", eid,ID, t_max, v_max)
                BESTP_E[eid].append([ID, t_max, v_max])
                if i > 10:
                    break

        IEP = self.VDB.IndividualProfiles

        from mvlib.plots import plot_profile

        plotdir = "plots_video_profiles_V2"
        os.mkdir(f"{plotdir}")

        for eid in BESTP_E:

            if os.path.exists(f"{plotdir}/{self.E.mapE[eid]}"):
                os.rmdir(f"{plotdir}/{self.E.mapE[eid]}")
                os.mkdir(f"{plotdir}/{self.E.mapE[eid]}")
            else:
                os.mkdir(f"{plotdir}/{self.E.mapE[eid]}")
            
            for [ID, t_max, v_max] in BESTP_E[eid]:
                print("BESTP_E", eid, ID, t_max, v_max, V_Durations[ID])
                Profile = get_video_profile_per_emotion(ID, IEP[ID], V_Durations[ID])

                for ti in Profile[eid]:
                    print("Profile", ID, eid, ti, Profile[eid][ti])

                mapE = self.E.mapE.copy()
                title = f"{mapE[eid]} "
                file = f"{ID}"
                
                plot_profile(Profile[eid], t_max, title, file, xlabel="time", ylabel="emotion level", fontsize=16, plotdir=f"{plotdir}/{self.E.mapE[eid]}")
                #exit()

        import json
        with open(f"{plotdir}/BESTP_E", 'w') as f:
            json.dump(BESTP_E, f, indent=4)

        return BESTP_E


    def clicks_videos(self,V_Durations):
        #self.IndividualProfiles[VID] = [t,eid]
        IEP = self.VDB.IndividualProfiles

        clicks_distribution, clicks_distribution_peruser, clicks_distribution_peruser_persecond = {},{},{}
        for e in range(9):
            clicks_distribution[e], clicks_distribution_peruser[e], clicks_distribution_peruser_persecond[e] = Counter(), Counter(), Counter()

        N = len(IEP)
        print("N", N)
        #exit()

        for VID in IEP:
            clicksE, clicks_peruserE, clicks_peruser_persecondE = clicks_per_emotion(IEP[VID], V_Durations[VID])
            print(VID, clicksE, len(IEP[VID]), V_Durations[VID])


            for e in range(0, 9):
                clicks_distribution[e][clicksE[e]] += 1
                clicks_distribution_peruser[e][clicks_peruserE[e]] += 1
                clicks_distribution_peruser_persecond[e][clicks_peruser_persecondE[e]] += 1

            #print("clikcs_per_user", clikcs_per_user, clikcs_per_user_persecond)




        from mvlib.plots import plot_video_clics
        mapE = self.E.mapE.copy()
        mapE[0] = "All emotions"
        for e in range(9):

            title = f"{mapE[e]}"
            file = f"ClicksVideoDistr_{mapE[e]}"
            if e == 0 :
                xmax = 500
            else:
                xmax = 100
            plot_video_clics(clicks_distribution[e], N,  title, file, xfactor = 1, xmax=xmax, xlabel="Number of Clicks",
                             ylabel="Percentage of videos", fontsize=16, plotdir=None)

            continue

            title = f"{mapE[e]}"
            file = f"ClicksVideoDistr_peruser_{mapE[e]}"
            plot_video_clics(clicks_distribution_peruser[e], N, title, file, xfactor = 100, xmax=None, xlabel="Number of Clicks per User",
                             ylabel="Percentage of videos",
                             fontsize=16, plotdir=None)



            title = f"{mapE[e]}"
            file = f"ClicksVideoDistr_peruser_persecond_{mapE[e]}"
            plot_video_clics(clicks_distribution_peruser_persecond[e], N, title, file,  xfactor = 10000, xmax=1, xlabel="Number of Clicks per User per 100 seconds",
                         ylabel="Percentage of videos",
                         fontsize=16, plotdir=None)

    def clicks_figure7(self, V_Durations):
        # self.IndividualProfiles[VID] = [t,eid]
        IEP = self.VDB.IndividualProfiles


        from mvlib.plots import plot_clicks_per_video_per_user_per_second
        from mvlib.plots import  plot_clicks_per_video_per_user_per_second_All_vs_Neutral
        import statistics as stat

        clicks_per_video_per_user_per_second = {}
        clicks = {}
        for EID in range(1, 10):
            clicks[EID] = []
            clicks_per_video_per_user_per_second[EID] = []

        Time_Scale = 100

        for VID in IEP:
            TVID = V_Durations[VID]
                #print(VID,TVID)
                #exit()
            i_profile_clicks = Counter()
            count_profile = 0
            for i_profile in IEP[VID]:
                count_profile +=1
                for [t, eid] in i_profile:

                    if t == 0:
                        if eid == 6: continue

                    for EID in range(1, 9):

                        if EID == eid:
                            i_profile_clicks[EID] += 1
                    if eid != 6:
                        i_profile_clicks[9] += 1

                #print(VID, count_profile, i_profile_clicks)

            for EID in range(1, 10):
                i_profile_clicks[EID] /= count_profile
                i_profile_clicks[EID] /= TVID
                i_profile_clicks[EID] *= Time_Scale

                #print(VID,EID, count_profile, i_profile_clicks[EID])
                #
                clicks[EID].append(i_profile_clicks[EID])
                # calculate standard deviation of list
            #exit()

        for EID in range(1, 10):
            avg = stat.mean(clicks[EID])
            sd = stat.stdev(clicks[EID])
            print("EID", EID , avg, sd)
            clicks_per_video_per_user_per_second[EID].extend([avg,sd])

        file = f"clicks_per_video_per_user_per_second_T{Time_Scale}"

        self.E.mapE[9] = f"Emotions"
        plot_clicks_per_video_per_user_per_second(clicks_per_video_per_user_per_second, self.E.mapE, file, xlabel=None, ylabel="Clicks per video", fontsize=16,
                    plotdir="plots_video_nov22")

        file = f"clicks_per_video_per_user_per_second_All_vs_Neutal_T{Time_Scale}"

        self.E.mapE[9] = f"Emotions"
        plot_clicks_per_video_per_user_per_second_All_vs_Neutral(clicks_per_video_per_user_per_second, self.E.mapE, file, xlabel=None,
                                                  ylabel="Clicks per video", fontsize=16,
                                                  plotdir="plots_video_nov22")

        exit()


    def clicks_stat(self):
        #self.IndividualProfiles[VID] = [t,eid]
        IEP = self.VDB.IndividualProfiles

        from mvlib.plots import plot_clicks

        clicks = Counter()
        for VID in IEP:
            for i_profile in IEP[VID]:
                i_profile_clicks = 0
                for [t, eid] in i_profile:

                    if t == 0:
                        if eid == 6:
                            continue
                    i_profile_clicks += 1
                clicks[i_profile_clicks] += 1

            V = 0
            for v in clicks:
                V += clicks[v]
            for v in clicks:
                clicks[v] = (clicks[v] / V) * 100

        file = f"clicks_all_emotions"

        self.E.mapE[0] =  f"All emotions"
        plot_clicks(clicks, 0, self.E, file, xlabel="Number of Clicks", ylabel="Percentage of profiles", fontsize=16,
                    plotdir="plots_video_oct22")

        for EID in range(1, 9):
            clicks = Counter()
            for VID in IEP:
                for i_profile in IEP[VID]:
                    i_profile_clicks = 0
                    for [t, eid] in i_profile:

                        if t == 0:
                            if eid == 6: continue

                        if EID == eid:
                            i_profile_clicks += 1

                    clicks[i_profile_clicks] += 1




            V = 0
            for v in clicks:
                V += clicks[v]
            for v in clicks:
                clicks[v] = (clicks[v]/V) * 100


            file = f"clicks_{self.E.mapE[EID]}"
            plot_clicks(clicks, EID, self.E, file, xlabel="Number of Clicks", ylabel="Percentage of profiles", fontsize=12, plotdir="plots_video_oct22")

    def aggregated_profiles_meansd(self):

        APTV = self.VDB.APTV

        APE_mean, APE_std = {}, {}
        for eid in range(9):
            APE_mean[eid] = []
            APE_std[eid] = []

        for t in range(120):
            print("t", t)
            for eid in range(1,9):
                p = []
                for VID in APTV:

                    T = len(APTV[VID][eid])
                    if t >= T: continue

                    eval = APTV[VID][eid][t]
                    p.append(eval)
                    # print(eid, VID, T, eval)

                APE_mean[eid].append(statistics.mean(p) )
                APE_std[eid].append(statistics.stdev(p) )
                # if t > 28 and t < 30:
                #    emotion_name = E.mapE[eid]
                #    floatlist2distribution(emotion_name, t, p, plotdir="plots/signal_distribution")

            from mvlib.plots import plot_aggregatedprofiles_meansd

            for eid in range(1,9):
                file = f"{self.E.mapE[eid]}_meansd"
                #print("APE_mean[eid]\n", APE_mean[eid])
                #print(" APE_std[eid]\n", APE_std[eid])
                plot_aggregatedprofiles_meansd(APE_mean[eid], APE_std[eid], eid, self.E, file, fontsize=12,
                                   plotdir="plots/aggregatedprofiles_meansd")

            #exit()

#CER distribution
    def CER_distribution(self,clip_length,jump):

        #self.VDB.get_dAPTV(clip_length)
        # self.dAPTV, self.dAPTV_stat = {}, {}
        # dAPTV[ID][t] = APTV[ID][emotion_id][t + clip_length] - APTV[ID][emotion_id][t]
        # stat[round(dAPTV[ID][t], 3)] += 1
        #self.VDB.get_dAPTV_porogs(self, jump, type="top")
        #self.porogs[eid]

        statINT = {}
        for eid in self.VDB.dAPTV_stat:
            statINT[eid] = Counter()
            norm = 0
            for v in self.VDB.dAPTV_stat[eid]:
                statINT[eid][int(v*100)] += self.VDB.dAPTV_stat[eid][v]
                norm += self.VDB.dAPTV_stat[eid][v]


            for v in sorted(statINT[eid]):
                statINT[eid][v] =  100* (statINT[eid][v] / norm)
                print(v,statINT[eid][v])

            from mvlib.plots import plot_CER
            top = self.VDB.porogs[eid]*100
            file = f"{self.E.mapE[eid]}_CER"
            plot_CER(statINT[eid], top, eid, self.E, file, xlabel="jump (percentage of users)", ylabel="percentage of cases", fontsize=16, plotdir="plots/CER/")

            #exit()

    def Signal2NoiseDistribution(self, clip_length):

        from mvlib.plots import plot_S2N
        from mvlib.Signal2noise import get_simulate_multiprocessing

        for eid in range(5, 9):

            fileS2N = f"{self.E.mapE[eid]}_S2N_data"

            S2N, step = get_simulate_multiprocessing(eid, clip_length, self.VDB)

            import json

            with open(fileS2N, 'w') as f:
                json.dump(S2N , f, indent=4)

            with open(fileS2N) as f:
                S2N = json.load(f)

            print("S2N", S2N)

            file = f"{self.E.mapE[eid]}_S2N"
            step =100/200
            plot_S2N(S2N, step, eid, self.E, file, xlabel="percentiles", ylabel="Signal-to-noise ratio", fontsize=12, plotdir="plots/Signal2Noise/")
            #exit()


class Video:

    def __init__(self, row):
        self.VID = re.search(rf".+/(.+?)\.mp4", row['Media']).group(1)
        self.AID = int(row['AdvertId'])
        self.Duration = int(row['Duration'])
        self.StarRating = float(row['DecimalStarRating'])
        self.MarketId = row['MarketId']
        self.Title = row['Title']
        # self.AdRecordCreatedDate = row['AdRecordCreatedDate']

    def add_video_path(self, video_dir_path):
        # /localstore/System1DataImport/jwripper/media/
        self.path_mp4 = f"{video_dir_path}/{self.VID}.mp4"

class VideoDB:

    def __init__(self, fileDescriptionVideos, fileIndividualProfiles,  dirVideos = None ):
        self.fileDescriptionVideos = fileDescriptionVideos
        self.fileIndividualProfiles = fileIndividualProfiles
        self.dirVideos = dirVideos


    def add_Emotions(self):
        self.Emotions = Emotions()

    def parseDescriptionVideos(self, market_filtr = None):

        self.VDB, self.VMAP = {}, {}
        with open(self.fileDescriptionVideos, 'r') as file:
            reader = csv.DictReader(file)

            for row in reader:
                V =Video(row)

                self.VDB[V.VID] = V
                self.VMAP[V.AID] = V.VID
                #print("self.VMAP",V.AID,V.VID)


    def filtrMarket(self, market_filtr=None):
        self.market_filtr = market_filtr
        statMarketId = Counter()
        VDBFiltered = {}
        for VID in self.VDB:
            V = self.VDB[VID]
            statMarketId[int(get_value(V.MarketId))] += 1

            if not self.market_filtr == None:
                if not int(get_value(V.MarketId)) in self.market_filtr:
                    continue
            VDBFiltered[V.VID] = self.VDB[V.VID]

        self.VDB = VDBFiltered
        print("statMarketId", statMarketId)
        print("self.VDB size:", len(self.VDB))
        self.get_aggregated_profiles()
        #exit()

    def split_train_validation_test(self, dir_save, rate = [80, 10, 10]):
        import random
        training, validation, test = [], [], []
        for VID in self.VDB:
            rv = random.uniform(0, 100)
            if rv < rate[0]:
                training.append(VID)
            elif rv < (rate[0] + rate[1]):
                validation.append(VID)
            else:
                test.append(VID)

        print("training, validation, test:", len(training), len(validation), len(test) )

        with open(f"{dir_save}/training", 'w') as f:
            for element in training:
                f.write(element + "\n")

        with open(f"{dir_save}/validation", 'w') as f:
            for element in validation:
                f.write(element + "\n")

        with open(f"{dir_save}/test", 'w') as f:
            for element in test:
                f.write(element + "\n")







    def add_video_dir_path(self,video_dir_path):
        self.dirVideos = video_dir_path
        for VID in self.VDB:
            self.VDB[VID].add_video_path(video_dir_path)

    def parseIndividualProfiles(self):

        ERRORs_no_AID , ERROR_EmotionString = {}, {}

        self.IndividualProfiles = {}
        with open(self.fileIndividualProfiles, 'r') as file:
            reader = csv.DictReader(file)

            for i, row in enumerate(reader):
                # print("row", i, row)
                #RID = int(get_value(row["'RespondentID'"]))
                try:
                    AID = int(get_value(row["'advertId'"]))
                except KeyError:
                    AID = int(get_value(row["AdvertId"]))
                except:
                    print("KeyError2 AdvertId")
                    exit()


                if AID not in self.VMAP:
                    ERRORs_no_AID[AID] = 1
                    #print("ERRORs_no_AID", AID)
                    continue

                try:
                    EmotionString = get_value(row["'EmotionString'"])
                except KeyError:
                    EmotionString = get_value(row["EmotionString"])
                except:
                    print("KeyError2 EmotionString")
                    exit()

                if "|" not in EmotionString:
                    ERROR_EmotionString[AID] = 1
                    #print("ERROR EmotionString", AID, EmotionString)
                    continue

                VID = self.VMAP[AID]
                if VID not in self.IndividualProfiles:self.IndividualProfiles[VID] = []
                #print("EmotionString", EmotionString)
                self.IndividualProfiles[VID].append(parse_emstring(EmotionString))


        print("ERRORs_no_AID", len(ERRORs_no_AID))
        print("ERROR_EmotionString", len(ERROR_EmotionString))

    def get_Durations(self):

        Duration = {}

        for VID in self.VDB:

            V = self.VDB[VID]
            T = V.Duration
            Duration[VID] = T

        return Duration

    def get_StarRate(self):

        StarRate = {}

        for VID in self.VDB:
            V = self.VDB[VID]
            SR = V.StarRating
            StarRate[VID] = SR

        return StarRate

    def get_aggregated_profiles(self):
        IEP = self.IndividualProfiles

        print("get_aggregated_profiles", len(self.VDB))

        self.APTV = {}
        for VID in self.VDB:
            APT = {}
            for eid in range(1,9):
                APT[eid] = Counter()

            V = self.VDB[VID]
            T = V.Duration

            i_users = len(IEP[VID])
            #print("T, i_users:", T, i_users)
            for i_profile in IEP[VID]:
                for [t, eid] in i_profile:
                    if t == 0:
                        if eid == 6:
                            continue

                    for ti in range(int(t) + 1, T):
                        APT[eid][ti] += 1

            for eid in APT:
                for ti in range(T):
                    APT[eid][ti] = round(APT[eid][ti] / i_users, 5)

            self.APTV[VID] = APT



    def get_dAPTV(self, clip_length):
        self.clip_length = clip_length
        self.dAPTV, self.dAPTV_stat = {}, {}
        self.max_Jump_Video = {}
        for eid in range(1, 9):
            self.dAPTV[eid], self.dAPTV_stat[eid], self.max_Jump_Video[eid] = get_dAPTV_EID(self.APTV, eid, clip_length)


    def get_percentile_ALL(self, eid, step=0.5):

        stat = self.dAPTV_stat[eid]
        V = 0
        for v in sorted(stat):
            V += stat[v]
        print("get_percentile_ALL", V, eid)


        p, vV = {}, 0
        for v in sorted(stat, reverse=True):
            print("v", v , stat[v] , vV)
            vV += stat[v] / V
            p[vV] = v



        porogs = []
        for i, s in enumerate(sorted(p, reverse=False)):
            if i == 0:
                sA, porogA = 0, p[s]
                #print("i==0;", porogA, sA )
                continue
            #print("i,s;", i, s, p[s])
            if s - sA > step/100:
                porogs.append([porogA, p[s], sA, s])
                porogA = p[s]
                sA = s
                #print("kk;", porogA, sA)

        #porogs.append([porogA, 0, sA, 1])
        print("porogs\n", porogs)
        #exit()
        return porogs


    def get_percentile(self, eid, porog, type="top"):

        stat = self.dAPTV_stat[eid]
        V = 0
        for v in sorted(stat):
            V += stat[v]
        print("get_percentile V", V, eid, porog)

        reverse = False
        if type == "top":
            reverse = True

        vV = 0
        for v in sorted(stat, reverse=reverse):
            print("v", v , stat[v] , vV)
            vV += stat[v] / V
            if v <= porog:
                p = vV
                print("get_percentile:", type, p, porog)
                break
        return p

    def get_porog(self, eid, jump, type="top"):

        stat = self.dAPTV_stat[eid]
        V = 0
        for v in sorted(stat):
            V += stat[v]
        # print("get_porog V", V, "jump", jump)

        reverse = False
        if type == "top":
            reverse = True

        vV = 0
        for v in sorted(stat, reverse=reverse):
            # print("v", v , stat[v] , vV)
            vV += stat[v] / V
            if vV > jump / 100:
                porog = v
                print("get_porog Threthhold:", type, porog, vV, V, int(vV * V))
                break
        return porog

    def get_dAPTV_porogs(self, jump, type="top"):
        self.jump = jump
        self.porogs = {}
        for eid in range(1, 9):
            self.porogs[eid]  = self.get_porog(eid, jump, type=type)
            #print(f"eid {eid} {type} {self.porogs[eid]}")

    def get_video_with_jump(self, jump):

        self.get_dAPTV_porogs( jump)
        self.video_with_jump = Counter()
        self.VID_with_jump = {}
        for eid in range(1, 9):

            # self.max_Jump_Video[eid][VID]
            for VID in self.max_Jump_Video[eid]:
                if self.max_Jump_Video[eid][VID] > self.porogs[eid]:
                    self.video_with_jump[eid] +=1
                    self.VID_with_jump[VID] = 1


            #print(f"eid {eid} {type} {self.porogs[eid]}")


    def get_dAPTV_porogs_fixed(self, porog_fixed):

        self.porogs = {}
        for eid in range(1, 9):
            self.porogs[eid]  = porog_fixed
            #print(f"eid {eid} {type} {self.porogs[eid]}")

    def filter_duplicate(self, array_ID):
        filter = {}
        array_ID_filtered = []
        for [VID, t, eid, dV] in array_ID:
            if t in filter:
                if dV > filter[t]:
                    filter[t] = dV
            else:
                filter[t] = dV
        filter2 = {}
        for [VID, t, eid, dV] in array_ID:
            if t in filter2: continue
            if filter[t] == dV:
                array_ID_filtered.append([VID, t, eid])
                filter2[t] = 1
        return array_ID_filtered

    def check_stat(self):
        count_DUPLICATE = 0
        positive_ID_filtr, stat = {}, Counter()
        for VID in self.positive_ID:
            for vt in self.positive_ID[VID]:
                keyV = f"{vt[0]}_{vt[1]}"
                if keyV in positive_ID_filtr:
                    count_DUPLICATE += 1

                positive_ID_filtr[keyV] = vt
                stat[vt[2]] += 1
        print("check_stat self.positive_ID count_DUPLICATE:", count_DUPLICATE)

        for eid in sorted(stat):
            print("check_stat eid:", eid, stat[eid])


    def get_positive_ID(self):

        self.positive_ID = {}
        for eid in self.dAPTV:
            for VID in self.dAPTV[eid]:
                for t in self.dAPTV[eid][VID]:
                    dV = self.dAPTV[eid][VID][t]
                    if dV < self.porogs[eid]: continue

                    if VID in self.positive_ID:
                        self.positive_ID[VID].append([VID, t, eid, dV])
                        self.positive_ID_filtr[f"{VID}_{t}"] = eid
                    else:
                        self.positive_ID[VID] = []
                        self.positive_ID[VID].append([VID, t, eid, dV])
                        self.positive_ID_filtr[f"{VID}_{t}"] = 1

        print("positive_ID:", len(self.positive_ID))

    def filtr_positive_ID(self):
        self.check_stat()
        for VID in self.positive_ID:
            self.positive_ID[VID] = self.filter_duplicate(self.positive_ID[VID])
        self.check_stat()




    def get_negative_ID(self):
        self.negative_ID = {}
        filtr = self.positive_ID_filtr
        for VID in self.dAPTV[5]:
            for t in self.dAPTV[5][VID]:
                keyF = f"{VID}_{t}"
                if keyF in filtr: continue

                if VID in self.negative_ID:
                    self.negative_ID[VID].append([VID, t, -1])
                else:
                    self.negative_ID[VID] = []
                    self.negative_ID[VID].append([VID, t, -1])

        print("negative_ID:", len(self.negative_ID))



    def parse_video_set(self, output_folder, mode ="frames"):
        error_video = {}
        list_videos = []
        for VID in self.VDB:
            fname = f"{self.dirVideos}/{VID}.mp4"
            if os.path.isfile(fname):
                list_videos.append([VID, fname])
            else:
                error_video[VID] =1

        print("error_video", len(error_video))
        print("list_videos", len(list_videos))

        if mode == "audio":
            parse_video_set(list_videos, output_folder, fps=None, mode="audio", SIZE_SPLIT=1000)

        if mode == "frames":
            parse_video_set(list_videos, output_folder, fps=10, mode="frames", SIZE_SPLIT=1000)












       

