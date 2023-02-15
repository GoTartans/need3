from tqdm import tqdm
import os
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np

from speechbrain.pretrained import SepformerSeparation as separator
import nemo.collections.asr as nemo_asr
from transformers import AutoTokenizer, AutoModelWithLMHead

from scipy.io.wavfile import write
import json

from kafka import KafkaConsumer, KafkaProducer

#info of the instance where kafka cluster is located
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
    bootstrap_servers = [EXTERNAL_IP+':'+PORT],
    # auto_offset_reset='earliest',
    # enable_auto_commit=False
    )

producer = KafkaProducer(
    bootstrap_servers=[EXTERNAL_IP+':'+PORT],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)


def softmax(x):
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("DEVICE = ", device)

    # 1) Denoise model(sepformer)
    denoise_model = separator.from_hparams(source = "speechbrain/sepformer-wham16k-enhancement", 
                                           savedir = None)
    denoise_model = denoise_model.to(device) # set by device
    denoise_model.device = device # assign the device

    # 2) Asr model(conformer)
    asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/stt_en_conformer_transducer_large")
    asr_model = asr_model.to(device)

    # 3) Sentiment model
    senti_tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")
    senti_model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-emotion")
    emotion_ids = {"sadness":24784, "joy":3922, "love":333, "anger":11213, "fear":2971, "surprise":4158}
       
    
    denoise_model.eval()
    asr_model.eval()
    senti_model.eval()


    print("Ready for the speech!!")
    
    # Kafka consumer
    data_path = 'myfile.wav'

    for message in consumer:
        incoming = message.value #.decode('utf-8')
        
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

            # step1: denoise
            d_start = time.time()
            est_source = denoise_model.forward(batch)
            d_end = time.time()

            # step2: ASR
            asr_input = est_source[:,:,0].to(device)
            asr_input_length = torch.tensor(est_source.shape[1]).unsqueeze(0).long().to(device)

            # step2-1: encoding in ASR
            a_start = time.time()
            encoded, encoded_len = asr_model.forward(input_signal = asr_input,
                                                     input_signal_length = asr_input_length)
            # step2-2: decoding in ASR
            best_hyp_text, _ = asr_model.decoding.rnnt_decoder_predictions_tensor(
                                                                        encoder_output=encoded,
                                                                        encoded_lengths=encoded_len,
                                                                        return_hypotheses=False
                                                                        )
            a_end = time.time()
            
            # step3 SA
            input_ids = senti_tokenizer.encode(best_hyp_text[0] + '</s>', return_tensors='pt')
            
            s_start = time.time()
            out = senti_model.forward(input_ids, decoder_input_ids = torch.LongTensor([[0]]))
            s_end = time.time()

            logits = out.logits.squeeze()
            emotion_raw_logits = {emotion: logits[value].item() for emotion, value in emotion_ids.items()}
            emotion_raw_logits = dict(sorted(emotion_raw_logits.items(), key=lambda item: item[1]))

            # Result1: logit & first/second sentiment
            emotion_softmax_logit = dict(zip(emotion_raw_logits.keys(), softmax(list(emotion_raw_logits.values()))))
            first_sentiment = list(emotion_raw_logits.keys())[-1]
            second_sentiment = list(emotion_raw_logits.keys())[-2]

            # Result2: Time
            d_time = round(d_end - d_start, 4)
            a_time = round(a_end - a_start, 4)
            s_time = round(s_end - s_start, 4)
            end2end_time = round(s_end - d_start, 4)
            Time = {"d_time" : d_time, "a_time" : a_time, "s_time":s_time, "end2end_time": end2end_time}

            # Report
            print("Recognized Text", best_hyp_text)
            # print("emotion_raw_logits", emotion_raw_logits)
            # print("emotion_softmax_logit", emotion_softmax_logit)
            print("first_sentiment", first_sentiment)
            print("second_sentiment", second_sentiment)
            print("Time", Time)

        Time['k_start'] = time.time()
        producer.send(TOPIC_NAME_PRO,
            key= b'senti',
            value={'text':best_hyp_text, 'emotion_softmax_logit': emotion_softmax_logit, 'first_emotion':first_sentiment,'second_sentiment':second_sentiment, 'Time':Time}
        )
        producer.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type = str, default = "/home/jiin/WORKING/need3/Denoise_ASR_Sentiment/myfile.wav")

    args = parser.parse_args()
        
    print(f"[Running Start] args = {args}")
    main(args)
    print(f"\n[Running Finish] args = {args}")