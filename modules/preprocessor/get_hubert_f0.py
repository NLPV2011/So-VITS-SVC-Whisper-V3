import argparse
import logging
import math
import multiprocessing
import os
from glob import glob
from random import shuffle

import torch
from tqdm import tqdm

import utils
from modules.mel_processing import spectrogram_torch

logging.getLogger("numba").setLevel(logging.WARNING)
import librosa
import numpy as np

hps = utils.get_hparams_from_file("configs/config.json")
sampling_rate = hps.data.sampling_rate
hop_length = hps.data.hop_length


def process_one(filename, hmodel):
    # print(filename)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wav, sr = librosa.load(filename, sr=sampling_rate)
    soft_path = filename + ".soft.pt"
    if not os.path.exists(soft_path):
        wav16k = librosa.resample(wav, orig_sr=sampling_rate, target_sr=16000)
        wav16k = torch.from_numpy(wav16k).to(device)
        c = hmodel.encoder(wav16k)
        torch.save(c.cpu(), soft_path)

    f0_path = filename + ".f0.npy"
    if not os.path.exists(f0_path):
        f0 = utils.compute_f0_parselmouth(
            wav, sampling_rate=sampling_rate, hop_length=hop_length
        )
        np.save(f0_path, f0)

    spec_path = filename.replace(".wav", ".spec.pt")
    if not os.path.exists(spec_path):
        # Process spectrogram
        # The following code can't be replaced by torch.FloatTensor(wav)
        # because load_wav_to_torch return a tensor that need to be normalized

        audio, sr = utils.load_wav_to_torch(filename)
        if sr != hps.data.sampling_rate:
            raise ValueError(
                "{} SR doesn't match target {} SR".format(
                    sr, hps.data.sampling_rate
                )
            )

        audio_norm = audio / hps.data.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)

        spec = spectrogram_torch(
            audio_norm,
            hps.data.filter_length,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            center=False,
        )
        spec = torch.squeeze(spec, 0)
        torch.save(spec, spec_path)


def process_batch(filenames):
    print("Loading speech encoder for content...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hmodel = utils.get_speech_encoder(device=device)
    print("Loaded speech encoder.")
    for filename in tqdm(filenames):
        process_one(filename, hmodel)


def get_hubert_f0(in_dir="dataset/44k"):
    filenames = glob(f"{in_dir}/*/*.wav", recursive=True)  # [:10]
    shuffle(filenames)
    multiprocessing.set_start_method("spawn", force=True)

    num_processes = 1
    chunk_size = int(math.ceil(len(filenames) / num_processes))
    chunks = [
        filenames[i: i + chunk_size] for i in range(0, len(filenames), chunk_size)
    ]
    print([len(c) for c in chunks], "chunks founded in dataset.")
    processes = [
        multiprocessing.Process(target=process_batch, args=(chunk,)) for chunk in chunks
    ]
    for p in processes:
        p.start()
