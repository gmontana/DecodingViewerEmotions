import os
from youtubesearchpython import VideosSearch
import urllib.request
from pytube import YouTube
import json
import shutil

import json
import argparse

def new_dir(path):
    if os.path.exists(path):
        print("exist:  ", path)

        return 1
    else:
        os.mkdir(path)
        return -1


def download_video( link , dir):
    try:
        # object creation using YouTube
        # which was imported in the beginning
        yt = YouTube(link)
    except:
        print("Connection Error")  # to handle exception


    mp4_files = yt.streams.filter(file_extension="mp4")
    mp4_369p_files = mp4_files.get_by_resolution("360p")
    mp4_369p_files.download(dir)
    print("Finished!!\n")


def main():

    global args

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    print("args", args)
    with open(args.config) as f:
        youtube_param = json.load(f)
    print("youtube_param", youtube_param)

    search_tags = youtube_param["youtube_search"]["search_tags"]

    youtube_folder = youtube_param["dataset"]["data_dir"]
    new_dir(youtube_folder)

    for search_tag in search_tags:
        search_tag_info = {}
        search_tag_id = search_tag.replace(" ", "_")
        search_tag_folder = f"{youtube_folder}/{search_tag_id}"
        new_dir(search_tag_folder)
        search_tag_folder_videos = f"{search_tag_folder}/videos"
        print("output_folder:", search_tag_folder_videos)
        new_dir(search_tag_folder_videos)

        videosSearch = VideosSearch(search_tag, limit=100)

        list_video = videosSearch.result()['result']
        for v in list_video:
            hms = v["duration"].split(":")
            if (len(hms) == 3):
                duration_seconds = sum(x * int(t) for x, t in zip([3600, 60, 1], v["duration"].split(":")))
            elif (len(hms) == 2):
                duration_seconds = sum(x * int(t) for x, t in zip([60, 1], v["duration"].split(":")))
            elif (len(hms) == 1):
                duration_seconds = sum(x * int(t) for x, t in zip([1], v["duration"].split(":")))
            else:
                print("Wrong duration format", v["duration"])
                exit(1)
            if duration_seconds > youtube_param["youtube_search"]["duration_limit"]:
                print(v['id'], v['title'], v["link"], v["duration"], duration_seconds)
                continue

            video_info = {
                "id": v['id'],
                "title": v['title'],
                "link": v["link"],
                "duration": v["duration"]
            }
            search_tag_info[v['id']] = video_info


            url_link = v["link"]

            download_video(url_link, "tmp")
            list_dir = os.listdir("tmp")
            for f in list_dir:
                if f.endswith("mp4"):
                    shutil.move(f"tmp/{f}", f'{search_tag_folder_videos}/{v["id"]}.mp4')

        with open(f'{search_tag_folder}/info.json', 'w') as f:
            json.dump(search_tag_info, f, indent=4)

        dataset_info = {}
        dataset_info["dataset"] = {}
        dataset_info["dataset"]["name"] = search_tag_id
        dataset_info["dataset"]["data_dir"] = search_tag_folder
        dataset_info["dataset"]["dir_videos"] = youtube_param["dataset"]["dir_videos"]
        dataset_info["dataset"]["dir_frames"] = youtube_param["dataset"]["dir_frames"]
        dataset_info["dataset"]["dir_audios"] = youtube_param["dataset"]["dir_audios"]
        dataset_info["dataset"]["fps"] = youtube_param["dataset"]["fps"]
        dataset_info["dataset"]["file_predict_list"] = f'{search_tag_folder}/{youtube_param["dataset"]["file_predict_list"]}'

        with open(f'{search_tag_folder}/dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=4)

        file_test = dataset_info["dataset"]["file_predict_list"]
        with open(f'{file_test}', "w") as f:
            for it in search_tag_info:
                id = search_tag_info[it]["id"]
                f.write(f"{id}\n")





if __name__ == '__main__':
    main()
