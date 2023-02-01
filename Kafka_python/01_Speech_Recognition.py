from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers=['34.132.166.200:9092']
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
            transcript = recognizer.recognize_google(audio) 
        except sr.RequestError:
            transcript = 'API unavailable'
        except sr.UnknownValueError:
            transcript = 'Waiting for input or speak more clearly'
        
        print(transcript)
    return(transcript)
    

run_app = True
i = 1

while run_app:
    print('Line - ' + str(i))
    output = recognize_speech_from_mic(r,mic)
    i = i + 1

    producer.send('test',
        key=b'voice', 
        value=bytes(output, 'utf-8')
    )

    if output == 'stop Voice': 
        run_app = False
    


