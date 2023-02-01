from pathlib import PurePath
from pydub import AudioSegment
import os
import audiosegment


# # wav를 원하는 hz로 resampling하는 코드
# for path, direct, files in os.walk("/home/ubuntu/LibriSpeech/test-clean/"):
    
#     for each_file in files:
#         if each_file.endswith('wav'):
#             file_path = os.path.join(path, each_file)
#             flac_tmp_audio_data = audiosegment.from_file(file_path).resample(sample_rate_Hz=16000, sample_width=2, channels=1)
#             new_each_file = each_file[:-5] + '.wav'
#             new_each_file_path = os.path.join(path, new_each_file)
#             flac_tmp_audio_data.export(new_each_file_path, format="wav")




# # flac 을 wav로 바꾸는 코드
# for path, direct, files in os.walk("/home/ubuntu/LibriSpeech/test-clean/"):
    
#     for each_file in files:
#         if each_file.endswith('flac'):
#             file_path = os.path.join(path, each_file)
#             flac_tmp_audio_data = AudioSegment.from_file(file_path, format='flac') # .resample(sample_rate_Hz=16000, sample_width=2, channels=1)
#             new_each_file = each_file[:-5] + '.wav'
#             new_each_file_path = os.path.join(path, new_each_file)
#             flac_tmp_audio_data.export(new_each_file_path, format="wav")


# flac 지우는 코드
# for path, direct, files in os.walk("/home/ubuntu/LibriSpeech/dev-other/"):
    
#     for each_file in files:
#         if each_file.endswith('flac'):
#             file_path = os.path.join(path, each_file)
#             if os.path.exists(file_path):
#                 os.remove(file_path)
