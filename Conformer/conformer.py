import nemo.collections.asr as nemo_asr
from pathlib import PurePath
from pydub import AudioSegment
import os
from evaluate import load
import pandas as pd 
import random
import time

asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/stt_en_conformer_transducer_large")
wer = load("wer")

label_path = '/home/ubuntu/LibriSpeech/test-other_label.csv' # change label's path if only inference label is not needed
label = pd.read_csv(label_path, on_bad_lines= 'skip', sep = ',')


labels_list = label.values.tolist()
current_dir = '/home/ubuntu/LibriSpeech/test-other/'


file_list = []
true_label_list = []


# label에 맞게 file을 입력받기 위한 코드 우리의 infernce 에선 필요 X
for label in labels_list:
    add_dir = label[0].split()[0]
    true_label_list.append(' '.join(label[0].split()[1:]))
    real_add_dir = '/'.join(add_dir.split('-')[:-1])
    each_file_dir = os.path.join(current_dir, real_add_dir, add_dir + '.wav')    
    file_list.append(each_file_dir)



start = time.time()
pred = asr_model.transcribe(file_list) # 모델 predict, 모델 입력은 list 형태로 들어가야함 
end = time.time()


wer_score = wer.compute(predictions=list(map(lambda x:x.upper(), pred[0])), references = true_label_list)
print('true wer', wer_score)

wer_score = wer.compute(predictions=list(map(lambda x:x.upper(), pred[1])), references = true_label_list)
print('second wer', wer_score)
print('operating time', (end - start) / len(labels_list))



# for path, direct, files in os.walk("/home/ubuntu/efs/final_project/LibriSpeech/dev-other"):
    
#     label = []

#     if len(files) > 0:

#         for each_file in files:
#             if each_file.endswith('txt'):
#                 file_path = os.path.join(path, each_file)
#                 f = open(file_path, "r")
#                 for x in f:
#                     label.append(x)


#         for each_file in files:

#             if each_file.endswith('wav'):
#                 file_path = os.path.join(path, each_file)
#                 wav_tmp_audio_data = [file_path]
#                 print(asr_model.transcribe(wav_tmp_audio_data))
