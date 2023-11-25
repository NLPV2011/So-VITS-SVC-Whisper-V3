# SFAVIOM - "Rewriting" The So-VITS-SVC =))))))))))))

## Updates
> Regarding the issue of **leaking of voice timbre**, there are still no statistics on this\
> Compared to [Diff-SVC](https://github.com/prophesier/diff-svc) and [So-VITS-SVC](https://github.com/svc-develop-team/so-vits-svc), Diff-SVC performs much better when the training data is of extremely high quality, but this repository can performs better on lower quality data sets. Additionally, this repository is much faster in inference speed than Diff-SVC but slightly slower than So-VITS-SVC but improves the model's pronunciation a lot.

## Model Overview
A singing voice conversion (SVC) model, using the Whisper PPG (Whisper Large V3) encoder to extract features from the input audio, sent into VITS along with the F0 to replace the original input to achieve a voice conversion effect with Mixed Decoder.

## Required models
+ Whisper Large V3：[large-v3](https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt)
  + Place under `pretrain`.
```shell
# For simple downloading.
# Whisper
wget -P pretrain/ https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt
```

## Dataset preparation
All that is required is that the data be put under the `dataset_raw` folder in the structure format provided below.
```shell
dataset_raw
├───speaker0
│   ├───xxx1-xxx1.wav
│   ├───...
│   └───Lxx-0xx8.wav
└───speaker1
    ├───xx2-0xxx2.wav
    ├───...
    └───xxx7-xxx007.wav
```

## Data pre-processing.
1. Resample

```shell
python resample.py
 ```
2. Preprocess
```shell
python preprocess.py
# Notice.
# The n_speakers value in the config will be set automatically according to the amount of speakers in the dataset.
# To reserve space for additionally added speakers in the dataset, the n_speakers value will be be set to twice the actual amount.
# If you want even more space for adding more data, you can edit the n_speakers value in the config after runing this step.
# This can not be changed after training starts.
```
After running the step above, the `dataset` folder will contain all the pre-processed data, you can delete the `dataset_raw` folder after that.

## Training.
```shell
python train.py -c configs/config.json -m 32k
```

## Inferencing.

Use [inference_main.py](inference_main.py)
+ Edit `model_path` to your newest checkpoint.
+ Place the input audio under the `raw` folder.
+ Change `clean_names` to the output file name.
+ Use `trans` to edit the pitch shifting amount (semitones). 
+ Change `spk_list` to the speaker name.

## Onnx Exporting.
### **When exporting Onnx, please make sure you re-clone the whole repository!!!**
Use [onnx_export.py](onnx_export.py)
+ Create a new folder called `checkpoints`.
+ Create a project folder in `checkpoints` folder with the desired name for your project, let's use `myproject` as example. Folder structure looks like `./checkpoints/myproject`.
+ Rename your model to `model.pth`, rename your config file to `config.json` then move them into `myproject` folder.
+ Modify [onnx_export.py](onnx_export.py) where `path = "NyaruTaffy"`, change `NyaruTaffy` to your project name, here it will be `path = "myproject"`.
+ Run [onnx_export.py](onnx_export.py)
+ Once it finished, a `model.onnx` will be generated in `myproject` folder, that's the model you just exported.
+ Notice: if you want to export a 48K model, please follow the instruction below or use `model_onnx_48k.py` directly.
    + Open [model_onnx.py](model_onnx.py) and change `hps={"sampling_rate": 32000...}` to `hps={"sampling_rate": 48000}` in class `SynthesizerTrn`.
    + Open [nvSTFT](/vdecoder/hifigan/nvSTFT.py) and replace all `32000` with `48000`
    ### Onnx Model UI Support
    + [MoeSS](https://github.com/NaruseMioShirakana/MoeSS)
+ All training function and transformation are removed, only if they are all removed you are actually using Onnx.

## Gradio (WebUI)
Use [sovits_gradio.py](sovits_gradio.py) to run Gradio WebUI
+ Create a new folder called `checkpoints`.
+ Create a project folder in `checkpoints` folder with the desired name for your project, let's use `myproject` as example. Folder structure looks like `./checkpoints/myproject`.
+ Rename your model to `model.pth`, rename your config file to `config.json` then move them into `myproject` folder.
+ Run [sovits_gradio.py](sovits_gradio.py)
