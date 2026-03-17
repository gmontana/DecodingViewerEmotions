"""
setup_data.py

Prepares data downloaded from HuggingFace for use with the TSAM codebase.

Usage:
    python setup_data.py --input ./adcumen-data

This script:
    1. Unzips 5-second_MP4_Clips.zip into a videos/ directory
    2. Extracts frames from each MP4 at 10fps into data/frames_fps_10/{clip_id}/
    3. Extracts audio from each MP4 into data/audios/{clip_id}.wav
    4. Generates split files (training_0, valid_0_p1, valid_0_p2) from CSVs
    5. Builds VDB.pickle from CSV labels
    6. Extracts backbone_weights.tar -> net_weigths/resnet50_miil_21k.pth
    7. Extracts tsam_weights.tar -> weights/
"""

import argparse
import csv
import os
import subprocess
import sys
import tarfile
import zipfile
import multiprocessing
from collections import Counter
from pathlib import Path


def extract_clips(input_dir, videos_dir):
    """Unzip 5-second_MP4_Clips.zip into videos_dir."""
    zip_path = os.path.join(input_dir, "5-second_MP4_Clips.zip")
    if not os.path.exists(zip_path):
        print(f"ERROR: {zip_path} not found")
        sys.exit(1)

    print(f"Extracting clips from {zip_path}...")
    os.makedirs(videos_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(videos_dir)
    print(f"Extracted clips to {videos_dir}")


def find_mp4_files(videos_dir):
    """Find all MP4 files in videos_dir (handles nested directories from zip)."""
    mp4_files = {}
    for root, dirs, files in os.walk(videos_dir):
        for f in files:
            if f.endswith(".mp4"):
                clip_name = f.replace(".mp4", "")
                mp4_files[clip_name] = os.path.join(root, f)
    return mp4_files


def extract_frames_worker(args):
    """Worker function for parallel frame extraction."""
    clip_name, mp4_path, frames_dir, fps = args
    out_dir = os.path.join(frames_dir, clip_name)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "ffmpeg", "-loglevel", "panic",
        "-i", mp4_path,
        "-vf", f"scale=-1:256,fps={fps}",
        "-q:v", "0",
        os.path.join(out_dir, "%06d.jpg")
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR extracting frames for {clip_name}: {e}")
        return clip_name
    return None


def extract_audio_worker(args):
    """Worker function for parallel audio extraction."""
    clip_name, mp4_path, audios_dir = args
    wav_path = os.path.join(audios_dir, f"{clip_name}.wav")

    cmd = [
        "ffmpeg", "-loglevel", "panic",
        "-i", mp4_path,
        "-y", wav_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR extracting audio for {clip_name}: {e}")
        return clip_name
    return None


def extract_frames(mp4_files, frames_dir, fps=10, workers=4):
    """Extract frames from all MP4 files at given fps."""
    os.makedirs(frames_dir, exist_ok=True)

    # Skip clips that already have frames
    tasks = []
    for clip_name, mp4_path in mp4_files.items():
        out_dir = os.path.join(frames_dir, clip_name)
        if os.path.isdir(out_dir) and len(os.listdir(out_dir)) > 0:
            continue
        tasks.append((clip_name, mp4_path, frames_dir, fps))

    if not tasks:
        print("All frames already extracted, skipping.")
        return []

    print(f"Extracting frames from {len(tasks)} clips using {workers} workers...")
    with multiprocessing.Pool(workers) as pool:
        errors = pool.map(extract_frames_worker, tasks)
    errors = [e for e in errors if e is not None]
    print(f"Frame extraction complete. Errors: {len(errors)}")
    return errors


def extract_audio(mp4_files, audios_dir, workers=4):
    """Extract audio from all MP4 files."""
    os.makedirs(audios_dir, exist_ok=True)

    tasks = []
    for clip_name, mp4_path in mp4_files.items():
        wav_path = os.path.join(audios_dir, f"{clip_name}.wav")
        if os.path.isfile(wav_path):
            continue
        tasks.append((clip_name, mp4_path, audios_dir))

    if not tasks:
        print("All audio already extracted, skipping.")
        return []

    print(f"Extracting audio from {len(tasks)} clips using {workers} workers...")
    with multiprocessing.Pool(workers) as pool:
        errors = pool.map(extract_audio_worker, tasks)
    errors = [e for e in errors if e is not None]
    print(f"Audio extraction complete. Errors: {len(errors)}")
    return errors


def read_csv(csv_path):
    """Read a HuggingFace CSV file and return list of dicts."""
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def generate_split_files(input_dir, output_dir):
    """Generate split files from HuggingFace CSVs.

    Split files contain unique Video_Name values (one per line).
    Returns dict mapping split_name -> list of CSV row dicts.
    """
    os.makedirs(output_dir, exist_ok=True)

    splits = {}
    csv_map = {
        "training.csv": ("training_0", "training"),
        "validation.csv": ("valid_0_p2", "validation"),
        "testing.csv": ("valid_0_p1", "test"),
    }

    for csv_name, (split_file, split_key) in csv_map.items():
        csv_path = os.path.join(input_dir, csv_name)
        if not os.path.exists(csv_path):
            print(f"WARNING: {csv_path} not found, skipping {split_key} split")
            continue

        rows = read_csv(csv_path)
        splits[split_key] = rows

        # Write split file: unique Video_Name values, one per line
        seen = set()
        split_path = os.path.join(output_dir, split_file)
        with open(split_path, 'w') as f:
            for row in rows:
                vid = row["Video_Name"]
                if vid not in seen:
                    seen.add(vid)
                    f.write(vid + "\n")

        print(f"Generated {split_path} with {len(seen)} unique videos ({len(rows)} clips)")

    return splits


def build_vdb(splits, output_path, clip_length=5, frames_dir=None):
    """Build VDB.pickle from CSV data.

    Creates a VideoDB-compatible pickle that works with the existing
    MultiJumpDataSet code. Videos are organized by Video_Name, with
    labeled clips at their original Start_Second offsets.

    CSV Label column contains numeric labels 0-7 mapping to:
        0=Anger, 1=Contempt, 2=Disgust, 3=Fear,
        4=Happiness, 5=Neutral, 6=Sadness, 7=Surprise
    These map to eid 1-8 (eid = label + 1).
    """
    from mvlib.mvideo_lib import VideoDB, Video
    from mvlib.utils import save_pickle

    emotion_list = ["Anger", "Contempt", "Disgust", "Fear",
                    "Happiness", "Neutral", "Sadness", "Surprise"]

    # Collect all clips grouped by Video_Name
    # video_labels[VID] = [(start_second, eid), ...]
    video_labels = {}
    total_clips = 0
    for split_name, rows in splits.items():
        for row in rows:
            vid = row["Video_Name"]
            label = int(row["Label"])
            eid = label + 1  # CSV uses 0-7, code uses 1-8
            start_sec = int(row["Start_Second"])

            if vid not in video_labels:
                video_labels[vid] = []
            video_labels[vid].append((start_sec, eid))
            total_clips += 1

    print(f"Building VDB from {total_clips} clips across {len(video_labels)} videos...")

    # Create VDB object without calling __init__
    vdb = VideoDB.__new__(VideoDB)
    vdb.fileDescriptionVideos = ""
    vdb.fileIndividualProfiles = ""
    vdb.dirVideos = None
    vdb.positive_ID_filtr = {}

    # Create Video objects for each unique Video_Name
    vdb.VDB = {}
    vdb.VMAP = {}
    for i, vid in enumerate(video_labels):
        # Estimate duration from max start_second + clip_length
        max_t = max(s for s, _ in video_labels[vid])
        duration = max_t + clip_length + 1

        # If frames directory exists, get actual frame count
        if frames_dir:
            vid_frames_dir = os.path.join(frames_dir, vid)
            if os.path.isdir(vid_frames_dir):
                n_frames = len(os.listdir(vid_frames_dir))
                duration = max(duration, n_frames // 10 + 1)

        v = Video.__new__(Video)
        v.VID = vid
        v.AID = i
        v.Duration = duration
        v.StarRating = 0.0
        v.MarketId = "826"
        v.Title = ""
        vdb.VDB[vid] = v
        vdb.VMAP[i] = vid

    # Build synthetic APTV (Aggregated Profile per Time point per Video)
    # For each video, create APTV entries covering all time points.
    # For each labeled clip at (start_sec, eid):
    #   APTV[VID][eid][start_sec] = 0, APTV[VID][eid][start_sec + clip_length] = 1.0
    # This makes get_dAPTV produce dAPTV[eid][VID][start_sec] = 1.0
    vdb.APTV = {}
    vdb.IndividualProfiles = {}
    for vid, labels in video_labels.items():
        duration = vdb.VDB[vid].Duration
        vdb.APTV[vid] = {}
        for eid in range(1, 9):
            vdb.APTV[vid][eid] = Counter()
            for t in range(duration + 1):
                vdb.APTV[vid][eid][t] = 0

        # Set labeled emotions to have jumps at their time offsets
        for start_sec, eid in labels:
            t_end = start_sec + clip_length
            if t_end <= duration:
                vdb.APTV[vid][eid][t_end] = 1.0

        # Minimal individual profiles (not used in inference)
        vdb.IndividualProfiles[vid] = []

    vdb.add_Emotions()

    save_pickle(output_path, vdb)
    print(f"Saved VDB.pickle to {output_path}")

    # Print label distribution
    label_counts = Counter()
    for labels in video_labels.values():
        for _, eid in labels:
            label_counts[eid] += 1
    print("Label distribution:")
    for i, name in enumerate(emotion_list):
        eid = i + 1
        print(f"  {name} (eid={eid}): {label_counts.get(eid, 0)}")


def extract_weights(input_dir, repo_root):
    """Extract backbone and TSAM weights from tar files."""

    # Backbone weights
    backbone_tar = os.path.join(input_dir, "backbone_weights.tar")
    backbone_dir = os.path.join(repo_root, "net_weigths")
    if os.path.exists(backbone_tar):
        os.makedirs(backbone_dir, exist_ok=True)
        print(f"Extracting backbone weights...")
        with tarfile.open(backbone_tar, 'r') as tf:
            tf.extractall(backbone_dir)
        # Find the .pth file and ensure it's at the expected path
        expected_path = os.path.join(backbone_dir, "resnet50_miil_21k.pth")
        if not os.path.exists(expected_path):
            # Search for .pth file in extracted contents
            for root, dirs, files in os.walk(backbone_dir):
                for f in files:
                    if f.endswith(".pth"):
                        src = os.path.join(root, f)
                        os.rename(src, expected_path)
                        print(f"Moved {src} -> {expected_path}")
                        break
        if os.path.exists(expected_path):
            print(f"Backbone weights at {expected_path}")
        else:
            print("WARNING: Could not find backbone .pth file in tar")
    else:
        print(f"WARNING: {backbone_tar} not found, skipping backbone weights")

    # TSAM weights
    tsam_tar = os.path.join(input_dir, "tsam_weights.tar")
    weights_dir = os.path.join(repo_root, "weights")
    if os.path.exists(tsam_tar):
        os.makedirs(weights_dir, exist_ok=True)
        print(f"Extracting TSAM weights...")
        with tarfile.open(tsam_tar, 'r') as tf:
            tf.extractall(weights_dir)

        # Ensure checkpoint directory structure exists
        checkpoint_dir = os.path.join(weights_dir, "checkpoint")
        expected_ckpt = os.path.join(checkpoint_dir, "balanced.ckpt.pth.tar")

        if not os.path.exists(expected_ckpt):
            os.makedirs(checkpoint_dir, exist_ok=True)
            # Search for checkpoint file
            for root, dirs, files in os.walk(weights_dir):
                for f in files:
                    if "balanced" in f and f.endswith(".pth.tar"):
                        src = os.path.join(root, f)
                        if src != expected_ckpt:
                            os.rename(src, expected_ckpt)
                            print(f"Moved {src} -> {expected_ckpt}")
                        break
                    elif f.endswith(".pth.tar") or f.endswith(".pth"):
                        src = os.path.join(root, f)
                        if src != expected_ckpt:
                            os.rename(src, expected_ckpt)
                            print(f"Moved {src} -> {expected_ckpt}")
                        break

        # Check for args.json
        args_json = os.path.join(weights_dir, "args.json")
        if not os.path.exists(args_json):
            # Search in extracted files
            for root, dirs, files in os.walk(weights_dir):
                for f in files:
                    if f == "args.json":
                        src = os.path.join(root, f)
                        if src != args_json:
                            os.rename(src, args_json)
                            print(f"Moved {src} -> {args_json}")
                        break

        if not os.path.exists(args_json):
            # Generate args.json with default TSM architecture config
            import json
            default_args = {
                "emotion_jumps": {
                    "emotion_ids": [1, 2, 3, 4, 5, 6, 7, 8],
                    "clip_length": 5,
                    "jump": 0.5,
                    "porog": 0.2,
                    "background_size": -1
                },
                "dataset": {
                    "name": "adcumen",
                    "data_dir": "./data",
                    "dir_videos": "videos",
                    "dir_frames": "frames_fps_10",
                    "dir_audios": "audios",
                    "fileVDB": "DataAdcumen/VDB.pickle",
                    "file_train_list": "DataAdcumen/training_0",
                    "file_val_list": "DataAdcumen/valid_0_p2",
                    "file_test_list": "DataAdcumen/valid_0_p1",
                    "video_img_param": {
                        "image_tmpl": "{:06d}.jpg",
                        "img_input_size": 256,
                        "img_output_size": 224
                    },
                    "video_augmentation": {
                        "RandomHorizontalFlip": True,
                        "scales": [1, 0.875, 0.75, 0.66],
                        "Adjust_sharpness": 2.0,
                        "ColorJitter": False,
                        "RandomGrayscale": 0.0,
                        "GaussianBlur": False
                    },
                    "audio_img_param": {
                        "window_sizes": [25, 50, 100],
                        "hop_sizes": [10, 25, 50],
                        "n_mels": 224,
                        "eps": 1e-6,
                        "spec_size": [3, 224, 224],
                        "num_segments": 1,
                        "m_segments": 1
                    },
                    "audio_augmentation": {
                        "status": True,
                        "random_shift_waveform": [0.1, 0.1]
                    },
                    "fps": 10
                },
                "TSM": {
                    "video_segments": 12,
                    "audio_segments": 1,
                    "motion": False,
                    "num_class": 8,
                    "main": {
                        "arch": "resnet50_timm",
                        "pretrain": "imagenet",
                        "dropout": 0.5,
                        "last_pool": 1,
                        "input_mode": 2,
                        "backbone_weights": "net_weigths/resnet50_miil_21k.pth"
                    },
                    "shift_temporal": {
                        "status": True,
                        "f_div": 8,
                        "shift_depth": 1,
                        "n_insert": 2,
                        "m_insert": 0
                    },
                    "shift_temporal_modality": {
                        "status": False,
                        "f_div": 8,
                        "n_insert": 2,
                        "m_insert": 1
                    },
                    "shift_spatial": {
                        "status": False,
                        "f_div": 8,
                        "n_insert": 2,
                        "m_insert": 1
                    },
                    "motion_param": {
                        "k_frames": 5,
                        "sharpen_cycles": 1,
                        "HW_conv_kernel": 9,
                        "HW_conv_sigma": 1.1,
                        "normadd": 0
                    }
                },
                "net_run_param": {
                    "epochs": 10,
                    "batch_size": 8,
                    "num_workers": 4
                },
                "net_optim_param": {
                    "lr": 0.1,
                    "lr_decay": [0.1, 3, 0.01, 6, 0.001, 8],
                    "momentum": 0.9,
                    "gd": 20,
                    "weight_decay": 1e-4
                },
                "save_epoch": [],
                "root_folder": "logs"
            }
            with open(args_json, 'w') as f:
                json.dump(default_args, f, indent=4)
            print(f"Generated {args_json}")

        if os.path.exists(expected_ckpt):
            print(f"TSAM weights at {expected_ckpt}")
        else:
            print("WARNING: Could not find TSAM checkpoint in tar")
    else:
        print(f"WARNING: {tsam_tar} not found, skipping TSAM weights")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare HuggingFace data for use with TSAM codebase")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to HuggingFace download directory")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Output directory for processed data (default: ./data)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers for frame/audio extraction")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames per second for frame extraction (default: 10)")
    parser.add_argument("--skip-frames", action="store_true",
                        help="Skip frame extraction")
    parser.add_argument("--skip-audio", action="store_true",
                        help="Skip audio extraction")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input)
    data_dir = os.path.abspath(args.data_dir)
    repo_root = os.path.dirname(os.path.abspath(__file__))

    print(f"Input directory: {input_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Repository root: {repo_root}")
    print()

    # Step 1: Extract video clips from zip (if available)
    videos_dir = os.path.join(data_dir, "videos")
    zip_path = os.path.join(input_dir, "5-second_MP4_Clips.zip")
    frames_dir = os.path.join(data_dir, "frames_fps_10")
    audios_dir = os.path.join(data_dir, "audios")

    if os.path.exists(zip_path):
        extract_clips(input_dir, videos_dir)
        print()

        # Find all MP4 files
        mp4_files = find_mp4_files(videos_dir)
        print(f"Found {len(mp4_files)} MP4 clips")
        print()

        # Step 2: Extract frames
        if not args.skip_frames:
            extract_frames(mp4_files, frames_dir, fps=args.fps, workers=args.workers)
        print()

        # Step 3: Extract audio
        if not args.skip_audio:
            extract_audio(mp4_files, audios_dir, workers=args.workers)
        print()
    else:
        print(f"NOTE: {zip_path} not found, skipping video/frame/audio extraction.")
        print("If you already have frames and audio extracted, set --data-dir to point there.")
        print()

    # Step 4: Generate split files from CSVs
    data_adcumen_dir = os.path.join(repo_root, "DataAdcumen")
    splits = generate_split_files(input_dir, data_adcumen_dir)
    print()

    # Step 5: Build VDB.pickle
    if splits:
        vdb_path = os.path.join(data_adcumen_dir, "VDB.pickle")
        build_vdb(splits, vdb_path, frames_dir=frames_dir if os.path.isdir(frames_dir) else None)
    print()

    # Step 6: Extract weights
    extract_weights(input_dir, repo_root)
    print()

    # Summary
    print("=" * 60)
    print("Setup complete!")
    print()
    print("To run inference on the test set:")
    print()
    print("  python predict.py \\")
    print("    --data config/default.json \\")
    print("    --model weights \\")
    print("    --type test \\")
    print("    --id test_run")
    print()
    print("Results will be saved to ./data/predicted/test_run/")
    print("=" * 60)


if __name__ == "__main__":
    main()
