from tqdm import tqdm

import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from ERC_dataset import STT_loader
from utils import make_batch_roberta, make_batch_bert, make_batch_gpt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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



def inference_models(args, denoise_model, asr_model, senti_model, batch, device='cuda'):


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


    # step 3: sentiment analysis
    test_sentence = best_hyp_text[0]
    
    emodict = {0: "anger", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"} 
    dataclass = args.cls
    DATA_loader = STT_loader
    make_batch = make_batch_roberta
    test_dataset = DATA_loader(test_sentence, dataclass)
    # test_utt_list = test_dataset.get_utters()
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=make_batch)

    start_sentiment = time.time()
    test_pred_list, inference_time, softmax_logits = Prediction(senti_model, test_dataloader)
    test_pred_emo = list(map(lambda s:emodict[s], test_pred_list))

    test_pred = test_pred_emo[0]
    softmax_logits = softmax_logits[0].tolist()

    end_sentiment = time.time()



    # step4: calculate time
    denoise_time = end_denoise - start_time
    asr_time = end_asr - start_asr
    senti_time = end_sentiment - start_sentiment
    end2end_time = end_sentiment - start_time


            
    return best_hyp_text, test_pred, softmax_logits, [denoise_time, asr_time, senti_time, end2end_time]
