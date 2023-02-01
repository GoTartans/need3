from kafka import KafkaProducer

# info of the instance where kafka cluster is located
EXTERNAL_IP = '35.236.200.251'
PORT = '9092'
producer = KafkaProducer(
    bootstrap_servers=[EXTERNAL_IP+':'+PORT]
)

import speech_recognition as sr

r = sr.Recognizer()
#sr.Microphone.list_microphone_names()
mic = sr.Microphone()

# Derived from https://realpython.com/python-speech-recognition/
def recognize_speech_from_mic(recognizer, microphone):
    # check that recognizer and microphone arguments are appropriate type
    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` must be `Recognizer` instance")

    if not isinstance(microphone, sr.Microphone):
        raise TypeError("`microphone` must be `Microphone` instance")
    
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source,phrase_time_limit=3)
        
        try:
            audio_wav = audio.get_wav_data(convert_rate=microphone.SAMPLE_RATE)
        except sr.UnknownValueError:
            audio_wav = 'Waiting for input or speak more clearly'
        
        # try: 
        #     transcript = recognizer.recognize_google(audio) 
        # except sr.RequestError:
        #     transcript = 'API unavailable'
        # except sr.UnknownValueError:
        #     transcript = 'Waiting for input or speak more clearly'
        # print(audio_raw)
        
        # print(transcript)
        # print(audio_wav)
        # print(type(audio_wav))
        # print(microphone.SAMPLE_RATE)
    return(audio_wav)
    

run_app = True
i = 1

TOPIC_NAME = 'wav_test'
while run_app:
    print('Line - ' + str(i))
    output = recognize_speech_from_mic(r,mic)
    i = i + 1
    
    with open('/root/need3/Mood Lamp Python/myfile_wav.wav', mode='bx') as f:
        f.write(output)
    
    # print(output)
    with open('/root/need3/Mood Lamp Python/myfile_byte.wav', mode='bx') as f:
        f.write(bytes(output))

    producer.send(TOPIC_NAME,
        key=b'voice', 
        value=bytes(output)#, 'utf-8')
    )

    if output == 'stop Voice': 
        run_app = False
    


