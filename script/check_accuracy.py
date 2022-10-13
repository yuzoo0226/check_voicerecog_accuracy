import os
import glob

# whisper
import whisper

# vosk
import wave
import sys
import json      
from vosk import Model, KaldiRecognizer, SetLogLevel

# speechrecognition google speech recognition
import speech_recognition as sr
 
GP_SELECTED_MODEL = "small.en"
GP_BASE_PATH = "./wave_data/BITEC_wavedata/BITEC_name_car_near_valid/wo_nr/*"

class CheckAccuracy:
    def __init__(self, selected_model):
        print("loaded model is ", selected_model)
        self._model_whisper = whisper.load_model(selected_model)
        self._model_vosk = Model(lang="en-us")
        self._model_google = sr.Recognizer()
        print("loading complete!")
    
    def recog_by_vosk(self, path, return_all=False, show_all_result=False):
        wf = wave.open(path, "rb")
        rec = KaldiRecognizer(self._model_vosk, wf.getframerate())
        rec.SetWords(True)
        rec.SetPartialWords(True)
        result_flag = True

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                if show_all_result:
                    final_result = rec.Result()
                    result_flag = False
                else:
                    final_result = rec.Result()
                    result_flag = False
            else:
                pass

        if result_flag:
            final_result = rec.FinalResult()
            if show_all_result:
                # 信頼値なども含めたデータを表示する
                print(final_result)
        else:
            pass

        dict_json = json.loads(final_result)
        recog_txt = dict_json["text"]

        if return_all:
            return dict_json
        else:
            return recog_txt


    def recog_by_google(self, path, language="en", show_all_result=False, return_all=False):
        with sr.AudioFile(path) as source:
            audio = self._model_google.record(source)

        try:
            if language == "ja":
                text = self._model_google.recognize_google(audio, language='ja-JP')
            else:
                text = self._model_google.recognize_google(audio, language="en")
        except:
            print("Unknown value error", "cannot recognize")
            return False
    
        return text

    def calc_predict_accuracy(self, base_path):
        # 指定したパスに有るデータをすべて取得
        files = glob.glob(base_path)
        ground_truth_values = {}

        # ファイル名からgtを取得し，検証しやすいように辞書型変数に整形する
        for file_path in files:
            ground_truth_values[file_path] = os.path.splitext(os.path.basename(file_path))[0]

        # 検証
        for path, gt in ground_truth_values.items():
            print("[file]:", path, "[Ground Truth]:", gt)

            # Whisperによる認識
            print("\n"*2, "="*10, "whisper result", "="*10)
            result = self._model_whisper.transcribe(path, language="en")
            print("recognition result", result["text"])
            # 小文字で統一し，文頭に有るスペースは削除
            result_4check = (result["text"].lower()).lstrip()
            if gt.lower() == result_4check:
                print("correct")

            # voskによる認識
            print("\n"*2, "="*10, "vosk result", "="*10)
            recog_txt = self.recog_by_vosk(path)
            print(recog_txt)
            if gt.lower() == recog_txt.lower():
                print("correct")

            # google speech recognitionによる認識
            print("\n"*2, "="*10, "google result", "="*10)
            recog_txt = self.recog_by_google(path)
            # 認識不能だった場合の対処
            if recog_txt == False:
                pass
            else:
                print(recog_txt)
                if gt.lower() == recog_txt.lower():
                    print("correct")

if __name__ == "__main__":
    selected_model = GP_SELECTED_MODEL
    base_path = GP_BASE_PATH
    ca = CheckAccuracy(selected_model)
    ca.calc_predict_accuracy(base_path)

