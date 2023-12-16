
import os

def new_dir(path):
    if os.path.exists(path):
        print("exist:  ", path)

        return 1
    else:
        os.mkdir(path)
        return -1

video_folder = "tmp_video"
file_video = 'Anger_0.mp4'
folder_frames =  file_video.replace(".mp4", "_frames")
new_dir(f"{video_folder}/{folder_frames}")
fps = 10
cmd_frames = f"ffmpeg -loglevel panic -i {video_folder}/{file_video}  -vf \"scale=-1:256,fps={fps}\" -q:v 0 \"{video_folder}/{folder_frames}/%06d.jpg\" "
print(cmd_frames)
os.system(cmd_frames)

file_audio = file_video.replace(".mp4", ".wav")
cmd_audio = f"ffmpeg -loglevel panic -i {video_folder}/{file_video}   {video_folder}/{file_audio}"

print(cmd_audio)
os.system(cmd_audio)

cmd_assemble = f"ffmpeg -framerate 10 -i \"{video_folder}/{folder_frames}/%06d.jpg\" -i {video_folder}/{file_audio} -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" \"{video_folder}/output.mp4\" "

print(cmd_assemble)