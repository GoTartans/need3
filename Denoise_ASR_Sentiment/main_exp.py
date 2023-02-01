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
from ERC_dataset import STT_loader
from model import ERC_model
from utils import make_batch_roberta, make_batch_bert, make_batch_gpt
from torch.utils.data import Dataset, DataLoader
import pdb
from sklearn.metrics import precision_recall_fscore_support
import psutil


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device =  'cpu'


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


def main(args):



    # Denoise model(sepformer)
    denoise_model = separator.from_hparams(source = "speechbrain/sepformer-wham16k-enhancement", 
                                           savedir = None)
    denoise_model = denoise_model.to(device) # set by device
    denoise_model.device = device # assign the device

    # Asr model(conformer)
    asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/stt_en_conformer_transducer_large")
    asr_model = asr_model.to(device)

    # metric1: (WER)
    metric = WordErrorRate()
    total_wer_sen_score = 0

    # metric2: time
    total_denoise_time = 0
    total_asr_time = 0
    total_end2end_time = 0
    count = 0

    # data_path 
    data_path = "/home/ubuntu/LibriSpeech/test-other/5484/24317/5484-24317-0000.wav"

    # Data loading
    batch, batch_sample_rate = torchaudio.load(data_path) # ex) torch.Size([1, 166960]), 16000
    batch = batch.to(device)

    #inference
    denoise_model.eval()
    asr_model.eval()

    with torch.no_grad():
        
        start_time = time.time()

        if args.denoise:
            # step1: denoise
            est_source = denoise_model.forward(batch)
            end_denoise = time.time()

            # step2: ASR
            asr_input = est_source[:,:,0].to(device)
            asr_input_length = torch.tensor(est_source.shape[1]).unsqueeze(0).long().to(device)

        else:
            end_denoise = start_time
            asr_input = batch
            asr_input_length = torch.tensor(batch.shape[1]).unsqueeze(0).long().to(device)
        
        # step2-1: encoding in ASR
        start_asr = time.time()
        encoded, encoded_len = asr_model.forward(input_signal = asr_input,
                                                    input_signal_length = asr_input_length)
        
        # step2-2: decoding in ASR
        best_hyp_text, _ = asr_model.decoding.rnnt_decoder_predictions_tensor(
                                                                            encoder_output=encoded,
                                                                            encoded_lengths=encoded_len,
                                                                            return_hypotheses=False
                                                                            )
        end_asr = time.time()   

        # step3 Sentiment analysis 

        initial = args.initial
        model_type = args.pretrained
        dataclass = args.cls
        Sentiment_model_path = args.Sentiment_model_path
        dataType = 'multi'
        DATA_loader = STT_loader

        # load model
        if 'roberta' in model_type:
            make_batch = make_batch_roberta
        elif model_type == 'bert-large-uncased':
            make_batch = make_batch_bert
        else:
            make_batch = make_batch_gpt      
        freeze = args.freeze
        if freeze:
            freeze_type = 'freeze'
        else:
            freeze_type = 'no_freeze'    
        sample = args.sample
        if 'gpt2' in model_type:
            last = True
        else:
            last = False


        test_sentence = best_hyp_text[0]
        test_dataset = DATA_loader(test_sentence, dataclass)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=make_batch)

        emodict = {0: "anger", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"} 
        clsNum = len(emodict)      
        model = ERC_model(model_type, clsNum, last, freeze, initial, device)
        save_path = os.path.join(Sentiment_model_path, model_type, initial, freeze_type, dataclass, str(sample))
        modelfile = os.path.join(save_path, 'model.bin')
        model.load_state_dict(torch.load(modelfile,  map_location=torch.device('cpu')))
        model = model.to(device)  
        model.eval()           
        test_utt_list = test_dataset.get_utters()

        memory_usage('#2')

        start_sentiment = time.time()
        test_pred_list, inference_time, softmax_logits = Prediction(model, test_dataloader)
        test_pred_emo = list(map(lambda s:emodict[s], test_pred_list))

        test_pred = test_pred_emo[0]
        softmax_logits = softmax_logits[0].tolist()

        print(test_pred)     # output: str
        print(softmax_logits)   # output: list of softmax logits 

        end_sentiment = time.time()



        # step4: calculate time
        denoise_time = end_denoise - start_time
        asr_time = end_asr - start_asr
        senti_time = end_sentiment - start_sentiment
        end2end_time = end_sentiment - start_time

            

    print(f"[Testing Done!]")
    print(f"(Average)denoise_time = {denoise_time:.4f}, asr_time = {asr_time:.4f}, senti_time = {senti_time:.4f} , end2end_time = {end2end_time:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type = str, default = "/home/ubuntu/LibriSpeech/test-clean")
    parser.add_argument("--denoise", type = bool, default = True)
    parser.add_argument( "--pretrained", help = 'roberta-large', default = 'roberta-large')
    parser.add_argument('-dya', '--dyadic', action='store_true', help='dyadic conversation')
    parser.add_argument( "--cls", help = 'emotion or sentiment', default = 'emotion')
    parser.add_argument( "--initial", help = 'pretrained or scratch', default = 'pretrained')
    parser.add_argument('-fr', '--freeze', action='store_true', help='freezing PM')
    parser.add_argument( "--sample", type=float, help = "sampling trainign dataset", default = 1.0) # 
    parser.add_argument( "--Sentiment_model_path", type= str, default = "/home/ubuntu/LibriSpeech/need3/CoMPM/MELD_models") # 
    args = parser.parse_args()
    
    print(f"[Running Start] args = {args}")
    main(args)
    print(f"\n[Running Finish] args = {args}")