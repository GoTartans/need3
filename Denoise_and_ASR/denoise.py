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

def main(args):

    # Device
    device = "cuda:0" if torch.cuda.is_available else "cpu"
    print("device", device)

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


    # Data loading
    for f1 in sorted(os.listdir(args.data_path)):
        f1_path = os.path.join(args.data_path, f1)

        for f2 in sorted(os.listdir(f1_path)):
            f2_path = os.path.join(f1_path, f2)

            # label
            label_path = f"{f1}-{f2}.trans.txt"
            label_path = os.path.join(f2_path, label_path)

            with open(label_path) as f:
                lines = f.readlines()

            labels = {line.split()[0] : " ".join(line.split()[1:]) for line in lines} # ex) {1089-134686-0032 : HE IS CALLED AS YOU KNOW THE APOSTLE OF THE INDIES}

            # flac_file
            for flac_file in sorted(os.listdir(f2_path)):
                flac_path = os.path.join(f2_path, flac_file)
                flac_file_name = flac_file.split(".flac")[0]
                try:
                    label = labels[flac_file_name].lower()
                except:
                    continue
                
         
                # batch
                batch, batch_sample_rate = torchaudio.load(flac_path) # ex) torch.Size([1, 166960]), 16000
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

                    # step3: calculate WER
                    wer_sen_score = metric(*best_hyp_text, label).item()

                    # step4: calculate time
                    denoise_time = end_denoise - start_time
                    asr_time = end_asr - start_asr
                    end2end_time = end_asr - start_time

                    # step5: report the result
                    print(f"\n[Result for {flac_file_name}]")
                    print(f"Wer_sen_score = {wer_sen_score:.4f}")
                    print(f"denoise_time = {denoise_time:.4f}, asr_time = {asr_time:.4f}, end2end_time = {end2end_time:.4f}")
                    print(f"pred = {best_hyp_text[0]}\nlabel = {label}")

                    # step6: accumulate it
                    total_wer_sen_score += wer_sen_score
                    total_denoise_time += denoise_time
                    total_asr_time += asr_time
                    total_end2end_time += end2end_time
                    count += 1
            
    print(f"[Testing Done!]")
    print(f"(Average)WER_Error = {total_wer_sen_score/count:.4f}")
    print(f"(Average)denoise_time = {total_denoise_time/count:.4f}, asr_time = {total_asr_time/count:.4f}, end2end_time = {total_end2end_time/count:.4f}")
    print(f"total_num_file = {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", 
                        type = str, 
                        default = "/data/gyuseok/LibriSpeech/test-clean")
    
    parser.add_argument("--denoise",
                        type = bool,
                        default = False)

    args = parser.parse_args()
    
    print(f"[Running Start] args = {args}")
    main(args)
    print(f"\n[Running Finish] args = {args}")