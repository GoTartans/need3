from tqdm import tqdm
from speechbrain.pretrained import SepformerSeparation as separator
import nemo.collections.asr as nemo_asr
import torchaudio
import os
import argparse
import time
from torchmetrics import WordErrorRate
import torch
import numpy as np
import copy
import logging
import random
import pandas as pd
import torch.nn as nn
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
# from ERC_dataset import STT_loader
from model import ERC_model
# from utils import make_batch_roberta, make_batch_bert, make_batch_gpt
from torch.utils.data import Dataset, DataLoader
import pdb
# from sklearn.metrics import precision_recall_fscore_support
import psutil

import io
import wave
import scipy.io.wavfile
import soundfile as sf
from scipy.io.wavfile import write
import json

from inference import inference_models
from kafka import KafkaConsumer, KafkaProducer


# info of the instance where kafka cluster is located
EXTERNAL_IPs = {
    'team3-gpu':'35.184.175.239',
    'instance-team3':'34.145.236.124',
    'Chanwoo':'34.132.166.200',
}
# EXTERNAL_IP = EXTERNAL_IPs['team3-gpu']
EXTERNAL_IP = 'localhost'
PORT = '9092'
TOPIC_NAME_CON = 'wav_test'
TOPIC_NAME_PRO = 'senti_test'

consumer = KafkaConsumer(
    TOPIC_NAME_CON,
    bootstrap_servers = [EXTERNAL_IP+':'+PORT]
    )

producer = KafkaProducer(
    bootstrap_servers=[EXTERNAL_IP+':'+PORT],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device =  'cpu'


def memory_usage(message: str = 'debug'):
    # current process RAM usage
    p = psutil.Process()
    rss = p.memory_info().rss / 2 ** 20 # Bytes to MB
    print(f"[{message}] memory usage: {rss: 10.5f} MB")


def Prediction(model, dataloader):
    model.eval()
    label_list = []
    pred_list = []
    start_time = time.time()
    m = nn.Softmax(dim=1)

    # label arragne
    with torch.no_grad():
        for i_batch, data in enumerate(tqdm(dataloader)):
            """Prediction"""
            batch_input_tokens, batch_speaker_tokens = data
            batch_input_tokens, = batch_input_tokens.to(device)
            pred_logits = model(batch_input_tokens, batch_speaker_tokens) # (1, clsNum)
            
            """Calculation"""    
            pred_label = pred_logits.argmax(1).item()
            pred_list.append(pred_label)
            softmax_logits = m(pred_logits)


    inference_time = time.time() - start_time
    inference_time /= len(pred_list)
    
    return pred_list, inference_time, softmax_logits

def convert_bytearray_to_wav_ndarray(input_bytearray: bytes, sampling_rate=16000):
    bytes_wav = bytes()
    byte_io = io.BytesIO(bytes_wav)
    write(byte_io, sampling_rate, np.frombuffer(input_bytearray, dtype=np.int16))
    output_wav = byte_io.read()
    output, samplerate = sf.read(io.BytesIO(output_wav))
    return output, samplerate

def main(args):

    print("DEVICE = ", device)

    # Denoise model(sepformer)
    denoise_model = separator.from_hparams(source = "speechbrain/sepformer-wham16k-enhancement", 
                                           savedir = None)
    denoise_model = denoise_model.to(device) # set by device
    denoise_model.device = device # assign the device

    # Asr model(conformer)
    asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/stt_en_conformer_transducer_large")
    asr_model = asr_model.to(device)

    # sentiment model
    
    initial = args.initial
    model_type = args.pretrained
    dataclass = args.cls
    Sentiment_model_path = args.Sentiment_model_path
    dataType = 'multi'
    # DATA_loader = STT_loader
    # make_batch = make_batch_roberta
    freeze = args.freeze
    freeze_type = 'freeze'
    last = False
    
    emodict = {0: "anger", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"} 
    clsNum = len(emodict)      
    model = ERC_model(model_type, clsNum, last, freeze, initial, device)
    save_path = Sentiment_model_path
    modelfile = os.path.join(save_path, 'model.bin')
    model.load_state_dict(torch.load(modelfile,  map_location=torch.device('cpu')))
    model = model.to(device)  
    model.eval()           
    
    denoise_model.eval()
    asr_model.eval()

    # metric1: (WER)
    metric = WordErrorRate()
    total_wer_sen_score = 0

    # metric2: time
    total_denoise_time = 0
    total_asr_time = 0
    total_end2end_time = 0
    count = 0


    print("Ready for the speech!!")
    print(args.denoise)
    
    # Kafka consumer
    # data_path = "/home/ubuntu/LibriSpeech/test-other/5484/24317/5484-24317-0000.wav"
    # data_path = args.data_path
    data_path = '/home/jiin/WORKING/need3/Denoise_ASR_Sentiment/myfile.wav'
    for message in consumer:
        incoming = message.value#.decode('utf-8')
        
        # save byte to wav file
        with open(data_path, mode='bw') as f:
            f.write(incoming)
        
        print("message coming")

        # Data loading
        batch, batch_sample_rate = torchaudio.load(data_path) # ex) torch.Size([1, 166960]), 16000
        batch = torchaudio.functional.resample(batch, orig_freq=batch_sample_rate, new_freq=16000)
        batch = batch.to(device)

        # inference
        with torch.no_grad():        
            best_hyp_text, test_pred, softmax_logits, time_list = inference_models(args, denoise_model, asr_model, model, batch, device)
            denoise_time, asr_time, senti_time, end2end_time = time_list
            
            print("Speech to Text = ", best_hyp_text[0])
            print("Predicted Emotion = ", test_pred)     # output: str
            print("Predicted Logits = ", softmax_logits)   # output: list of softmax logits 

            print(f"[Testing Done!]")
            print(f"(Average)denoise_time = {denoise_time:.4f}, asr_time = {asr_time:.4f}, senti_time = {senti_time:.4f} , end2end_time = {end2end_time:.4f}")


        producer.send(TOPIC_NAME_PRO,
            key= b'senti',
            value={'emotion':test_pred,'logits':softmax_logits}
        )
        producer.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_path", type = str, default = "/home/ubuntu/LibriSpeech/test-clean")
    # parser.add_argument("--data_path", type = str, default = "/home/jiin/WORKING/need3/Kafka_python/myfile_wav.wav")
    parser.add_argument("--denoise", type = int, default = 1)
    parser.add_argument( "--pretrained", help = 'roberta-large', default = 'roberta-large')
    parser.add_argument('-dya', '--dyadic', action='store_true', help='dyadic conversation')
    parser.add_argument( "--cls", help = 'emotion or sentiment', default = 'emotion')
    parser.add_argument( "--initial", help = 'pretrained or scratch', default = 'pretrained')
    parser.add_argument('-fr', '--freeze', action='store_true', help='freezing PM')
    parser.add_argument( "--sample", type=float, help = "sampling trainign dataset", default = 1.0) # 
    parser.add_argument( "--Sentiment_model_path", type= str, default = "/home/jiin/WORKING/need3/Denoise_ASR_Sentiment/pretrained_models/MELD_models") # 
    args = parser.parse_args()
        
    print(f"[Running Start] args = {args}")
    main(args)
    print(f"\n[Running Finish] args = {args}")