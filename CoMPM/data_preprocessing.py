from nltk.tokenize import word_tokenize  
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

import os
import string

# f = open('dataset/MELD/multi/MELD_test.txt', 'r')
f = open('dataset/dailydialog/dailydialog_dev.txt', 'r')
dataset = f.readlines()
f.close()

# os.mkdir('dataset/MELD_preprocessed')
# f = open('dataset/MELD_preprocessed/MELD_preprocessed_test.txt', 'w')
# os.mkdir('dataset/dailydialog_preprocessed')
f = open('dataset/dailydialog_preprocessed/dailydialog_preprocessed_dev.txt', 'w')

for i, data in enumerate(dataset):
    if i < 2:
        f.write(data)
        continue
    if data == '\n':
        f.write('\n')
        continue
    speaker, utt, emo = data.strip().split('\t')
    
    token = word_tokenize(utt)
    result = []   # 빈 리스트를 만들어준다.

    for word in token:   # token에 있는 단어 불러오기
        if word not in stopwords.words('english'):   # 불러온 단어가 stopwords에 있는가?
            result.append(word)   # 없으면 result에 넣기
    
    preprocessed_utt = ' '.join(result)


    out = ''.join([j for j in preprocessed_utt if j not in string.punctuation])
    preprocessed_utt = out

    f.write('\t'.join([speaker, preprocessed_utt, emo]))
    f.write('\n')
  
f.close()

  