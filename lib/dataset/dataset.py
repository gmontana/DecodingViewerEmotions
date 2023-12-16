
from PIL import Image
import os

import torch

from collections import Counter



from lib.dataset.audio import get_audio_x
from lib.dataset.video import  get_video_x
from mvlib.utils import save_pickle, load_pickle
from mvlib.mvideo_lib import VideoDB, Video

import random




class MultiJumpDataSet(torch.utils.data.Dataset):
    def __init__(self, input_file,  args ={}, mode_train_val = "training", fixed_porogs = -1 ):

        print("MultiJumpDataSet!!!!!")

        self.input_file = input_file
        self.args = args
        self.mode_train_val = mode_train_val

        param_dataset = args["dataset"]
        self.path_video_imagefolders = f'{param_dataset["data_dir"]}/{param_dataset["dir_frames"]}'
        self.path_audio_folder = f'{param_dataset["data_dir"]}/{param_dataset["dir_audios"]}'
        self.fps = param_dataset["fps"]


        self.param_adcumen = args["emotion_jumps"]
        self.clip_length = self.param_adcumen["clip_length"]
        self.emotion_ids = self.param_adcumen["emotion_ids"]
        self.jump =  self.param_adcumen["jump"]
        self.negative_size =  self.param_adcumen["background_size"]


        fileVDB = f'{param_dataset["fileVDB"]}'
        self.VDB = load_pickle(fileVDB)
        self.VDB.add_Emotions()
        if "market_filtr" in  args:
            self.VDB.filtrMarket(market_filtr=args["market_filtr"])
        print("self.VDB size:", len(self.VDB.VDB))
        self.VDB.get_dAPTV(self.clip_length )

        if(fixed_porogs == -1): self.VDB.get_dAPTV_porogs(self.jump , type="top")
        if(fixed_porogs > 0): self.VDB.get_dAPTV_porogs_fixed(fixed_porogs)

        self.VDB.get_positive_ID()
        self.VDB.filtr_positive_ID()
        self.VDB.get_negative_ID()

        self.parse_input_file_negative()
        self.parse_input_file_positive()
        self.unite_positive_negative()

        #self.__getitem__(0)
        #exit()


    def __getitem__(self, index):

        record = self.records[index]
        x_video, x_audio = torch.zeros(1), torch.zeros(1)
        ID = record["ID"]
        y = int(record["label"])

        if self.args["video_segments"] > 0:
            x_video = get_video_x(record,  self.args, self.mode_train_val)

        if self.args["audio_segments"] > 0:
            x_audio = get_audio_x(record,  self.args, self.mode_train_val)

        #print("__getitem__", y , x_video.size())
        return  ID,   y , x_video, x_audio


    def __len__(self):
        return len(self.records)


    def parse_input_file_positive(self):

        ERROR = Counter()
        self.records_positive = []
        self.stat_eid_positive = Counter()
        #lines = [x.strip().split(' ') for x in open(self.input_file)]

        if self.input_file != None:
            lines = [x.strip().split(' ') for x in open(self.input_file)]
        else:
            lines = [[ID] for ID in os.listdir(self.path_video_imagefolders)]

        for i, item in enumerate(lines):
            ID = item[0]

            if ID in self.VDB.positive_ID:

                line_data_t = self.get_data_ID(self.VDB.positive_ID[ID], ID,ERROR)

                if not line_data_t ==  None:
                    for line_data in line_data_t:
                        line_data["ID"] = ID
                        self.stat_eid_positive[line_data["eid"]]  +=1

                self.records_positive.extend(line_data_t)
        print("self.records_positive", len(self.records_positive))
        print(ERROR)

    def parse_input_file_negative(self):

        ERROR = Counter()
        self.records_negative = []

        self.stat_eid_negative = Counter()

        if self.input_file != None:
            lines = [x.strip().split(' ') for x in open(self.input_file)]
        else:
            lines = [[ID] for ID in os.listdir(self.path_video_imagefolders)]


        #lines = [x.strip().split(' ') for x in open(self.input_file)]

        for i, item in enumerate(lines):
            ID = item[0]

            if ID in self.VDB.negative_ID:

                line_data_t = self.get_data_ID(self.VDB.negative_ID[ID], ID, ERROR)

                if not line_data_t == None:
                    for line_data in line_data_t:
                        line_data["ID"] = ID
                        self.stat_eid_negative[line_data["eid"]] += 1

                #print("self.records_negative:", len(self.records_negative))
                if not line_data_t == None:
                    self.records_negative.extend(line_data_t)
        print("self.records_negative", len(self.records_negative))
        print(ERROR)

    def unite_positive_negative(self):

        #max_key = max(self.stat_eid_negative, key=self.stat_eid_negative.get)
        #print("unite_positive_negative max_key:", max_key, self.stat_eid_negative[max_key])
        self.records = self.records_positive.copy()
        if self.negative_size > 0:
            self.records_negative_SAMPLE = random.sample(self.records_negative, self.negative_size)
            print("self.records", len(self.records))
            print("self.records_negative_SAMPLE", len(self.records_negative_SAMPLE))
            self.records.extend(self.records_negative_SAMPLE)
            print("self.records", len(self.records))

            self.stat_eid_positive[-1] = self.negative_size

        stat = Counter()
        self.map_eid, self.map_label = {}, {}
        #sorted(counter.items(),key = lambda i: i[0])
        for i, eid in enumerate(sorted(self.stat_eid_positive.items(),key = lambda i: i[0])):
            self.map_eid[eid[0]] = i
            self.map_label[i] = eid[0]
            #print("stat_eid", i, eid, self.stat_eid_positive[eid[0]])

        for line_data in self.records:
            label = self.map_eid[line_data["eid"]]
            line_data["label"] = label
            stat[label] += 1

        print("stat label", stat)
        print("self.records:", len(self.records))
        #exit()


    def get_data_ID(self,list_VID_t, ID,ERROR):
        #print("get_data_ID", list_VID_t, ID)

        line_data = {}
        if self.args["video_segments"] > 0:

            ifolder = f"{self.path_video_imagefolders}/{ID}"
            if not os.path.isdir(ifolder):
                print("WARNING not exist", ID)
                return None

            ifolder_size = int(len(os.listdir(ifolder)))
            if ifolder_size < 5:
                print("WARNING no frames", ID)
                ERROR["WARNING no frames"] += 1
                return None

            line_data["imagefolder"] = ifolder
            line_data["imagefolder_size"] = ifolder_size



        if self.args["audio_segments"] > 0:
            wavefile = f"{self.path_audio_folder}/{ID}.wav"
            if not os.path.isfile(wavefile):
                print("WARNING not wavefile", ID)
                ERROR["WARNING not wavefile"] +=1
                return None

            line_data["audio_file"] = wavefile

        line_data_t = []
        for [VID, t, eid] in list_VID_t:

            ## check folder size
            ifolder_size = line_data["imagefolder_size"]
            iprofile_size = self.fps * (t + self.clip_length)
            if iprofile_size > ifolder_size:
                #print("## check folder size", VID, t, iprofile_size, ifolder_size)
                ERROR["## check folder size"] +=1
                continue
            ## check folder size

            line_data_new = line_data.copy()
            line_data_new["t"] = t
            line_data_new["eid"] = eid
            #print("teid", t, eid)

            line_data_t.append( line_data_new)

        #print("line_data_t" , line_data_t)

        #exit()
        return line_data_t





def GetDataSet(args, mode_train_val="training", fixed_porogs = -1):

    param_dataset = args["dataset"]

    if mode_train_val == "training":
        file_list = f'{param_dataset["file_train_list"]}'
    elif mode_train_val == "validation":
        file_list = f'{param_dataset["file_val_list"]}'
    elif mode_train_val == "test":
        file_list = f'{param_dataset["file_test_list"]}'
    else:
        file_list = None

    MMD = MultiJumpDataSet(file_list, args=args, mode_train_val=mode_train_val, fixed_porogs = fixed_porogs )

    return MMD

def GetDataLoaders(MMD, args):
    shuffle = True
    #shuffle = False
    #if mode_train_val == "training": shuffle = True

    data_loader = torch.utils.data.DataLoader(
        MMD,
        batch_size=args["net_run_param"]["batch_size"],
        num_workers=args["net_run_param"]["num_workers"],
        shuffle=shuffle, pin_memory=True, drop_last=False
    )


    return data_loader


