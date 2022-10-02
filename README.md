# How to Use

## clone this repository
```
cd your/path
git clone --recursive https://github.com/yuzoo0226/whisper_ws.git
```

## docker build
```
cd whisper_ws
docker build -t whisper_withmodel docker/
docker run --gpus all -v /home/your/path/whisper_ws/:/workspace/whisper_ws -it whisper_withmodel
```

## use python
```
cd whisper_ws/wave_data
python3
```

## python command
```
import whisper
model = whisper.load_model("medium")
file_path = "./BITEC_wavedata/BITEC_name_car_near/test_wave00_nr.wav"
result = result = model.transcribe(file_path, language="en", verbose=True)
```