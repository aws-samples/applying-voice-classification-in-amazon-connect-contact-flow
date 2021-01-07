import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import boto3
import os
from six import BytesIO
import numpy as np
from coswara_model import NetM3

s3 = boto3.client('s3')
sm = boto3.client('sagemaker-runtime')
    
def model_fn(model_dir):    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NetM3()
    if torch.cuda.device_count() > 1:
        print("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


def input_fn(request_body, request_content_type):
    if request_content_type == 'text/csv':
        new_sr=8000
        audio_len=20
        sampling_ratio=5
        tmp=request_body[5:]
        bucket=tmp[:tmp.index('/')]
        print("bucket: {}".format(bucket))
        obj=tmp[tmp.index('/')+1:]
        print("object: {}".format(obj))
        s3.download_file(bucket, obj, '/audioinput.wav')
        print("audio input file size: {}".format(os.path.getsize('/audioinput.wav')))
        waveform, sample_rate = torchaudio.load('/audioinput.wav')
        waveform = torchaudio.transforms.Resample(sample_rate, new_sr)(waveform[0, :].view(1, -1))
        const_len = new_sr * audio_len
        tempData = torch.zeros([1, const_len])
        if waveform.shape[1] < const_len:
            tempData[0, : waveform.shape[1]] = waveform[:]
        else:
            tempData[0, :] = waveform[0, :const_len]
        sound = tempData
        tempData = torch.zeros([1, const_len])
        if sound.shape[1] < const_len:
            tempData[0, : sound.shape[1]] = sound[:]
        else:
            tempData[0, :] = sound[0, :const_len]
        sound = tempData
        new_const_len = const_len // sampling_ratio
        soundFormatted = torch.zeros([1, 1, new_const_len])
        soundFormatted[0, 0, :] = sound[0, ::5]
        return soundFormatted
    elif request_content_type in ['application/x-npy', 'application/python-pickle']:
        return torch.tensor(np.load(BytesIO(request_body), allow_pickle=True))
    else:
        print("unknown request content type: {}".format(request_content_type))
        return request_body

    
def predict_fn(input_data, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    print("I am in predict_fn")
    with torch.no_grad():
        return model(input_data.to(device))
    