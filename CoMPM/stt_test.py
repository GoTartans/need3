# -*- coding: utf-8 -*-
from tqdm import tqdm
import os
import random
import pandas as pd
import time
import torch
import torch.nn as nn

from transformers import RobertaTokenizer
# from ERC_dataset import MELD_loader, Emory_loader, IEMOCAP_loader, DD_loader
from ERC_dataset import STT_loader
from model import ERC_model
from utils import make_batch_roberta, make_batch_bert, make_batch_gpt

from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
import pdb
import argparse, logging
from sklearn.metrics import precision_recall_fscore_support
import psutil
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def memory_usage(message: str = 'debug'):
    # current process RAM usage
    p = psutil.Process()
    rss = p.memory_info().rss / 2 ** 20 # Bytes to MB
    print(f"[{message}] memory usage: {rss: 10.5f} MB")

## finetune RoBETa-large
def main():    
    total_start_time = time.time()
    initial = args.initial
    model_type = args.pretrained
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
    
    """Dataset Loading"""
    dataset_list = ['STT']
    DATA_loader_list = [STT_loader]
    dataclass = args.cls
    dataType = 'multi'
    
    
    """Log"""
    log_path = os.path.join('STT_test.log')
    fileHandler = logging.FileHandler(log_path)
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)    
    logger.setLevel(level=logging.DEBUG)  
    
    memory_usage('#1')  
    
    """Model Loading"""
    for dataset, DATA_loader in zip(dataset_list, DATA_loader_list):
        if dataset == 'MELD':
            data_path = os.path.join('dataset', dataset, dataType)
        else:
            data_path = os.path.join('dataset', dataset)
        if 'preprocessed' in dataset:
            model_dataset = dataset[:-13]
        else:
            model_dataset = dataset
        save_path = os.path.join('MELD'+'_models', model_type, initial, freeze_type, dataclass, str(sample))
        print("###Save Path### ", save_path)
    
        # dev_path = os.path.join(data_path, dataset+'_dev.txt')
        test_path = os.path.join(data_path, dataset+'_test.txt')

        test_dataset = DATA_loader(test_path, dataclass)
        # for data in test_dataset:
        #     print(data)
            
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=make_batch)
        
        # dataset_all = []        
        # for idx in range(len(dialogs)):
        #     data = make_batch([(dialogs[idx], labelList, sentidict)])
        #     dataset_all.append(data)       
            
        # print(dataset_all[0])
        
        
        print('Data: ', dataset, '!!!')
        # clsNum = len(dev_dataset.labelList) 
        emodict = {0: "anger", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"} 
        clsNum = len(emodict)      
        model = ERC_model(model_type, clsNum, last, freeze, initial, device)
        modelfile = os.path.join(save_path, 'model.bin')
        model.load_state_dict(torch.load(modelfile))
        model = model.to(device)  
        model.eval()           
        test_utt_list = test_dataset.get_utters()

        """Dev & Test evaluation"""
        logger.info('####### ' + dataset + ' #######')
        
        memory_usage('#2')
        
        test_pred_list, inference_time = Prediction(model, test_dataloader)
        test_pred_emo = list(map(lambda s:emodict[s], test_pred_list))

        # print in stt_test.log file
        logger.info('inference time per utterance = {}'.format(inference_time))
        total_time = time.time() - total_start_time
        logger.info('total test time = {}'.format(total_time))
        for utt, pred in zip(test_utt_list, test_pred_emo):
            logger.info('{} \t {}'.format(utt, pred))
        logger.info('')

        test_df =  pd.DataFrame([test_utt_list, test_pred_emo], index=['utterance', 'pred']).T
        test_df.to_csv('results/{}_test.csv'.format(dataset))
        
        memory_usage('#3')
    
def Prediction(model, dataloader):
    model.eval()
    # correct = 0
    label_list = []
    pred_list = []
    start_time = time.time()
    
    # label arragne
    with torch.no_grad():
        for i_batch, data in enumerate(tqdm(dataloader)):
            """Prediction"""
            batch_input_tokens, batch_labels, batch_speaker_tokens = data
            batch_input_tokens, batch_labels = batch_input_tokens.to(device), batch_labels.to(device)
            
            pred_logits = model(batch_input_tokens, batch_speaker_tokens) # (1, clsNum)
            
            """Calculation"""    
            pred_label = pred_logits.argmax(1).item()
            # true_label = batch_labels.item()
            
            pred_list.append(pred_label)
            # label_list.append(true_label)
        #     if pred_label == true_label:
        #         correct += 1
        # acc = correct/len(dataloader)
    # return acc, pred_list, label_list
    inference_time = time.time() - start_time
    inference_time /= len(pred_list)
    
    return pred_list, inference_time

if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "Emotion Classifier" )    
    parser.add_argument( "--pretrained", help = 'roberta-large', default = 'roberta-large')
    parser.add_argument('-dya', '--dyadic', action='store_true', help='dyadic conversation')
    parser.add_argument( "--cls", help = 'emotion or sentiment', default = 'emotion')
    parser.add_argument( "--initial", help = 'pretrained or scratch', default = 'pretrained')
    parser.add_argument('-fr', '--freeze', action='store_true', help='freezing PM')
    parser.add_argument( "--sample", type=float, help = "sampling trainign dataset", default = 1.0) # 
        
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    main()
    