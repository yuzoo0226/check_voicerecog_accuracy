# How to Use

## clone this repository
```
cd your/path
git clone --recursive https://github.com/yuzoo0226/check_voicerecog_accuracy.git
```

## docker build
```
cd check_voicerecog_accuracy
docker build -t whisper_withmodel docker/
docker run --gpus all -v /home/your/path/check_voicerecog_accuracy/:/workspace/check_voicerecog_accuracy -it whisper_withmodel
```

## use python
```
cd check_voicerecog_accuracy/wave_data
python3
```

## python command
```
import whisper
model = whisper.load_model("medium")
file_path = "./BITEC_wavedata/BITEC_name_car_near/test_wave00_nr.wav"
result = model.transcribe(file_path, language="en", verbose=True)
```