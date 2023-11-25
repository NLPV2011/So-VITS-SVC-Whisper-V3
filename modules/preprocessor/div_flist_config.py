import argparse
import json
import os
import re
import wave
from random import shuffle

from tqdm import tqdm


def get_wav_duration(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        # get audio frames
        n_frames = wav_file.getnframes()
        # get sampling rate
        framerate = wav_file.getframerate()
        # calculate duration in seconds
        duration = n_frames / float(framerate)
    return duration

def div_flist_config(conf, train_list="./filelists/train.txt", val_list="./filelists/val.txt", source_dir="./dataset/44k"):
    pattern = re.compile(r'^[\.a-zA-Z0-9_\/]+$')

    train = []
    val = []
    idx = 0
    spk_dict = {}
    spk_id = 0
    for speaker in tqdm(os.listdir(source_dir)):
        spk_dict[speaker] = spk_id
        spk_id += 1
        wavs = ["/".join([source_dir, speaker, i]) for i in os.listdir(os.path.join(source_dir, speaker))]
        new_wavs = []
        for file in wavs:
            if not file.endswith("wav"):
                continue
            if not pattern.match(file):
                print(
                    f"Warning: The file name of {file} contains non-alphanumeric and underscores, which may cause issues. (or maybe not)")
            if get_wav_duration(file) < 0.3:
                print("skip too short audio:", file)
                continue
            new_wavs.append(file)
        wavs = new_wavs
        shuffle(wavs)
        train += wavs[2:]
        val += wavs[:2]

    shuffle(train)
    shuffle(val)

    print("Writing", train_list)
    with open(train_list, "w") as f:
        for fname in tqdm(train):
            wavpath = fname
            f.write(wavpath + "\n")

    print("Writing", val_list)
    with open(val_list, "w") as f:
        for fname in tqdm(val):
            wavpath = fname
            f.write(wavpath + "\n")

    conf["spk"] = spk_dict
    conf["model"]["n_speakers"] = spk_id

    print("Writing configs/config.json")
    with open("configs/config.json", "w") as f:
        json.dump(conf, f, indent=2)
