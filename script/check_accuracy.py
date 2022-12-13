import os
import glob
import csv

# whisper
import whisper

# vosk
import wave
import sys
import json      
from vosk import Model, KaldiRecognizer, SetLogLevel
from GP_DICTIONARY import *

# speechrecognition google speech recognition
import speech_recognition as sr
 
GP_SELECTED_WHISPER_MODEL = "large"
GP_BASE_PATH = "./wave_data/BITEC_wavedata/BITEC_name_car_near/wo_nr/enhancement/*"
OUTPUT_FILENAME = "./outputs/result.csv"
DICTIONARY_TYPE = "name"

class CheckAccuracy:
    # モデルの読み込みを事前に行う
    def __init__(self, selected_model):
        print("loaded model is ", selected_model)
        self._model_whisper = whisper.load_model(selected_model)
        self._model_vosk = Model(lang="en-us")
        self._model_google = sr.Recognizer()
        self.dicts = self.load_dictionary()
        print("loading complete!")


    #  置換辞書で設定したとおりに、文字列の置換を行う
    def replace_txt(self, recog_txt, replace_dics):
        for replace_key, replace_lists in replace_dics.items():
            for replace_list in replace_lists:
                recog_txt = recog_txt.replace(replace_list, replace_key)

        return recog_txt

    def remove_other_txt(self, recog_txt):
        # 中間にある[unk]を弾く
        non_unk_txt = recog_txt.replace('[unk] ', '')
        # 末尾にある[unk]を弾く
        non_unk_txt = non_unk_txt.replace('[unk]', '')

        # 文字列からスペースを見つける
        pos = non_unk_txt.find(' ')

        if pos != -1:
            # スペースが見つかったときは、スペース以降の文字列を除く
            correct_txt = non_unk_txt[:pos]
        else:
            # スペースが見つからなかったときはそのまま出力する（スペースがない = すでに1単語のみである）
            correct_txt = non_unk_txt

        return correct_txt


    # pythonで作成した音声認識用の辞書をvoskで扱える形に整形する関数
    def load_dictionary(self):
        dicts = {}
        # 辞書型変数から音声認識用の辞書を作成
        for dict_name, dict_values in GP_DICTIONARY.items():
            temp_dictionary = "[\"" # 辞書作成開始，voskに合わせた整形

            for dict_value in dict_values:
                temp_dictionary = temp_dictionary + dict_value + " "

            temp_dictionary = temp_dictionary[:-1] + "\"]" # 最後のスペースを削除した上で，voskに合わせて整形
            temp_dictionary_unk = temp_dictionary[:-2] + "\", \"[unk]\"]" # unknownを追加した辞書も自動で作成

            # print(dict_name, temp_dictionary) # 確認用
            dict_name_with_unk = dict_name + "_unk"
            dicts[dict_name] = temp_dictionary # 作成した音声認識用の辞書を辞書型配列に保存
            dicts[dict_name_with_unk] = temp_dictionary_unk
        print(dicts) # 確認用
        return dicts

    def read_csv_array(self, file_path) -> "array":
        csv_arrays = []
        with open(file_path) as f:
            reader = csv.reader(f)
            for row in reader:
                csv_arrays.append(row)
        return csv_arrays


    # voskを用いた音声認識を行う関数
    def recog_by_vosk(self, path, dictionary_type=None, return_all=False, show_all_result=False):
        wf = wave.open(path, "rb")
        # 辞書指定が合った場合は，辞書を読み込む
        if dictionary_type != None:
            try:
                rec = KaldiRecognizer(self._model_vosk, wf.getframerate(), self.dicts[dictionary_type])
            except:
                # 指定された辞書が見つからなかった場合は，辞書無しでの認識を行う
                print("cannot use selected dictionary")
                rec = KaldiRecognizer(self._model_vosk, wf.getframerate())
        else:
            # 辞書指定がない場合（デフォルト）
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

    # SpeechRecognitionで用意されているgoogleの音声認識を用いて音声認識を行う
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


    # 複数の音声認識を行い，csvファイルに結果を出力する
    def calc_predict_accuracy(self, base_path, csv_filepath="temp.csv", show_all_result=False):

        # 指定したパスに有るデータをすべて取得
        files = glob.glob(base_path)
        ground_truth_values = {}

        try:
            csv_arrays = read_csv_array(csv_filepath)
            for array in csv_arrays:
                ground_truth_values[array[0]] = array[1]
        except:
            # ファイル名からgtを取得し，検証しやすいように辞書型変数に整形する
            for file_path in files:
                ground_truth_values[file_path] = os.path.splitext(os.path.basename(file_path))[0]

        # csvファイルの出力に使用するヘッダー
        headers = [ ["", "whisper", "", "vosk(without dict)", "", "vosk(with dict)", "", "google", ""], ["Ground Truth", "正誤判定", "認識結果", "正誤判定", "認識結果", "正誤判定", "認識結果", "正誤判定", "認識結果",] ]

        # 結果出力用のcsvファイルの準備
        with open(OUTPUT_FILENAME, "w") as f:
            writer = csv.writer(f)
            writer.writerows(headers)

            # 検証
            for path, gt in ground_truth_values.items():
                # csvに書き込むための配列
                csv_row = []
                print("[file]:", path, "[Ground Truth]:", gt)

                csv_row.append(gt)

                ##################################################
                # Whisperによる認識
                ##################################################
                result = self._model_whisper.transcribe(path, language="en")
                # 表示モードのときのみ，結果を表示
                if show_all_result:
                    print("\n"*2, "="*10, "whisper result", "="*10)
                    print("recognition result", result["text"])

                # 小文字で統一し，文頭に有るスペースは削除
                recog_txt_whisper = (result["text"].lower()).lstrip()

                # 認識結果が合っていた場合はtrue,間違っていた場合はfalseをcsvに書き込む
                if gt.lower() == recog_txt_whisper:
                    csv_row.append("true")
                else:
                    csv_row.append("false")

                # 認識結果も書き込む
                csv_row.append(recog_txt_whisper)


                ##################################################
                # voskによる認識
                # （辞書なし）
                ##################################################
                recog_txt_vosk = self.recog_by_vosk(path)
                # 結果を表示するモードのときのみ表示
                if show_all_result:
                    print("\n"*2, "="*10, "vosk result", "="*10)
                    print(recog_txt_vosk)

                # 認識結果が合っていた場合はtrue,間違っていた場合はfalseをcsvに書き込む
                if gt.lower() == recog_txt_vosk.lower():
                    csv_row.append("true")
                else:
                    csv_row.append("false")

                # 認識結果も書き込む
                csv_row.append(recog_txt_vosk)


                ##################################################
                # voskによる認識
                # 辞書あり
                ##################################################
                recog_txt_vosk = self.recog_by_vosk(path, dictionary_type=DICTIONARY_TYPE)
                # 結果を表示するモードのときのみ表示
                if show_all_result:
                    print("\n"*2, "="*10, "vosk result", "="*10)
                    print(recog_txt_vosk)

                # 認識結果が合っていた場合はtrue,間違っていた場合はfalseをcsvに書き込む
                if gt.lower() == recog_txt_vosk.lower():
                    csv_row.append("true")
                else:
                    csv_row.append("false")

                # 認識結果も書き込む
                csv_row.append(recog_txt_vosk)


                ##################################################
                # google speech recognitionによる認識
                ##################################################
                recog_txt_google = self.recog_by_google(path)
                if show_all_result:
                    print("\n"*2, "="*10, "google result", "="*10)
    
                # 認識不能だった場合の対処
                if recog_txt_google == False:
                    if show_all_result:
                        print("認識結果なし")
                    csv_row.append("false")
                    csv_row.append("認識結果なし")
                else:
                    if show_all_result:
                        print(recog_txt_google)

                    if gt.lower() == recog_txt_google.lower():
                        csv_row.append("true")
                    else:
                        csv_row.append("false")
                    csv_row.append(recog_txt_google)


                # csvファイルに書き込み
                writer.writerow(csv_row)

            # 全ファイルの認識終了
            f.close()

if __name__ == "__main__":
    selected_model = GP_SELECTED_WHISPER_MODEL
    base_path = GP_BASE_PATH
    ca = CheckAccuracy(selected_model)
    ca.calc_predict_accuracy(base_path)