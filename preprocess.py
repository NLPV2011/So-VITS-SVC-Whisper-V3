import argparse
from modules.preprocessor.div_flist_config import div_flist_config
from modules.preprocessor.get_hubert_f0 import get_hubert_f0

config_template = {
  "train": {
    "log_interval": 200,
    "eval_interval": 800,
    "seed": 1234,
    "epochs": 10000,
    "learning_rate": 0.0001,
    "betas": [
      0.8,
      0.99
    ],
    "eps": 1e-09,
    "batch_size": 6,
    "fp16_run": True,
    "lr_decay": 0.999875,
    "segment_size": 10240,
    "init_lr_ratio": 1,
    "warmup_epochs": 0,
    "c_mel": 45,
    "c_kl": 1.0,
    "use_sr": True,
    "max_speclen": 512,
    "port": "8001",
    "keep_ckpts": 3,
    "all_in_mem": False
  },
  "data": {
    "training_files": "filelists/train.txt",
    "validation_files": "filelists/val.txt",
    "max_wav_value": 32768.0,
    "sampling_rate": 48000,
    "filter_length": 2048,
    "hop_length": 512,
    "win_length": 2048,
    "n_mel_channels": 128,
    "mel_fmin": 0.0,
    "mel_fmax": None
  },
  "model": {
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 1280,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "p_dropout": 0.1,
    "resblock": "1",
    "resblock_kernel_sizes": [3,7,11],
    "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
    "upsample_rates": [ 8, 8, 2, 2, 2],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [16,16, 4, 4, 4],
    "n_layers_q": 3,
    "use_spectral_norm": False,
    "gin_channels": 1280,
    "ssl_dim": 1280,
    "n_speakers": 200
  },
  "spk": {
    "nyaru": 0,
    "huiyu": 1,
    "nen": 2,
    "paimon": 3,
    "yunhao": 4
  }
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", type=str, default="./filelists/train.txt", help="path to train list")
    parser.add_argument("--val_list", type=str, default="./filelists/val.txt", help="path to val list")
    parser.add_argument("--source_dir", type=str, default="./dataset/44k", help="path to source dir")
    parser.add_argument(
        "--in_dir", type=str, default="dataset/44k", help="path to input dir"
    )
    args = parser.parse_args()
    
    div_flist_config(conf=config_template, train_list=args.train_list, val_list=args.val_list, source_dir=args.source_dir)
    get_hubert_f0(in_dir=args.in_dir)