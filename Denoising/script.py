from tqdm import tqdm
from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio
import os


folder_path = 'merged'

clean_path = 'merged_clean'

if not os.path.isdir(clean_path):
    os.makedirs(clean_path)

# bring model here
model = separator.from_hparams(source="speechbrain/sepformer-wham16k-enhancement", savedir='pretrained_models/sepformer-wham16k-enhancement')
# for custom file, change path
name_list = os.listdir(folder_path)
for file_name in tqdm(name_list, total = len(name_list)):
    file_path = os.path.join(folder_path, file_name)
    # inference happenss here
    est_sources = model.separate_file(path=file_path) 

    new_path = os.path.join(clean_path, file_name)
    torchaudio.save(new_path, est_sources[:, :, 0].detach().cpu(), 16000)

