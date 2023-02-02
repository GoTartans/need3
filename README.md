# need3

## Pi5
    mic_to_kafka.py
        - speech_recognition 라이브러리 사용
        - 마이크로 음성을 입력, kafka의 'wav' 토픽으로 produce

## GCP instance
    kafka cluster
        - zookeeper 와 broker를 통해 활성화
        - 'wav'와 'senti' 두 가지 토픽 존재

    wav_to_senti.py
        - ASR, Denois, SA 모델 연결
        - input: kafka의 'wav' 토픽에서 consume
        - output: kafka의 'senti' 토픽으로 produce

## Pi6
    kafka_to_bulb.py
        - kasa 라이브러리 사용
        - kafka의 'senti' 토픽에서 consume, 전구로 빛을 출력