from kafka import KafkaProducer

# info of the instance where kafka cluster is located
EXTERNAL_IPs = {
    'team3-gpu':'35.184.175.239',
    'instance-team3':'34.145.236.124',
    'Chanwoo':'34.132.166.200',
}
EXTERNAL_IP = EXTERNAL_IPs['team3-gpu']
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
            audio_wav = None

    return(audio_wav)


run_app = True
i = 1

TOPIC_NAME = 'wav_test'
while run_app:
    output = recognize_speech_from_mic(r,mic)

    if output is not None:
        print('Line - ' + str(i))
        producer.send(TOPIC_NAME,
            key=b'voice', 
            value=bytes(output)#, 'utf-8')
        )

    # if output == 'stop Voice': 
    #     run_app = False
    
    i = i + 1