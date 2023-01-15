from google.cloud import speech
import os
import io
from torchmetrics import WordErrorRate
import argparse




def main(args):
    # GCP connect
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]  = "iitp-class-team-3-2dddf32f474e.json"

    # step1: setup
    config = speech.RecognitionConfig(
        encoding = speech.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz = 16000,
        language_code = "en-US"
    )
    client = speech.SpeechClient()
    metric = WordErrorRate() 

    # step2: inference
    predictions = []
    targets = []
    total_time = 0.0
    count = 0

    for f1 in sorted(os.listdir(args.data_path)):
        f1_path = os.path.join(args.data_path, f1)

        for f2 in sorted(os.listdir(f1_path)):
            f2_path = os.path.join(f1_path, f2)

            # label
            label_path = f"{f1}-{f2}.trans.txt"
            label_path = os.path.join(f2_path, label_path)

            with open(label_path) as f:
                lines = f.readlines()

            labels = {line.split()[0] : " ".join(line.split()[1:]) for line in lines}

            # speech
            for flac_file in sorted(os.listdir(f2_path)):
                flac_path = os.path.join(f2_path, flac_file)
                flac_file_name = flac_file.split(".flac")[0]
                try:
                    label = labels[flac_file_name].lower()
                except:
                    continue

                # read audio
                with io.open(flac_path, "rb") as audio_file:
                    content = audio_file.read()
                
                # preprocessing
                audio = speech.RecognitionAudio(content = content)

                # inference
                response = client.recognize(config = config, 
                                            audio = audio)
                
                # result
                try:
                    result = response.results[0]
                except:
                    print(f"[Error of {flac_file_name}] there is no result through")
                    continue
                pred = result.alternatives[0].transcript.lower()
                time = result.result_end_time
                sec  = float(time.total_seconds())

                # print
                wer_word_score = metric(pred, label) 
                print(f"[Result for {flac_file_name}]\nwer_word_score = {wer_word_score:.4f}, time = {sec}sec\npred = {pred}\nlabel = {label}")

                # save
                predictions.append(pred)
                targets.append(label)
                total_time += sec
                count += 1
    
    # testing
    wer_score = metric(predictions, targets)
    print(f"[Testing] wer_score = {wer_score:.4f}, total_time = {total_time:.4f}, each_word_time = {total_time/count :.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apikey_path", type = str, default = "iitp-class-team-3-2dddf32f474e.json", help = "path of json file for api key")
    parser.add_argument("--data_path", type = str, default = "/data/gyuseok/LibriSpeech/test-clean")

    args = parser.parse_args()
    main(args)