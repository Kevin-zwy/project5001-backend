
from rest_framework import viewsets, status
from rest_framework.response import Response 
from .models import AudioFile
from .serializers import AudioFileSerializer
from .isy5001_softvoting_implemention import extract_features,soft_voting,load_sklearn_model,load_model
import librosa
import numpy as np
import torch
import torch.nn.functional as F
import io
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import os


class AudioFileViewSet(viewsets.ModelViewSet):
    queryset = AudioFile.objects.all()
    serializer_class = AudioFileSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        #return Response({'result': "YES"}, status=status.HTTP_201_CREATED)
        if serializer.is_valid():
            serializer.save()

            # 获取上传的WAV文件�?�?
            audio_file = serializer.instance.audio_file
            audio_file_path = audio_file.path

               
               
               
               


            # 确保文件存在
            if os.path.exists(audio_file_path):
                # 使用 librosa 加载音频文件
                audio_data, sr = librosa.load(audio_file_path, sr=None)
                print("音频数据加载成功！")
            else:
                print("文件不存在，请检查路径。")


            features = extract_features(audio_data, sr)

            rf_model_path = r'D:\project\project5001-backend-master\myproject\detection\rf_classifier.joblib'
            knn_model_path = r'D:\project\project5001-backend-master\myproject\detection\knn_classifier.joblib'
            dnn_model_path = r'D:\project\project5001-backend-master\myproject\detection\dnn_model (1).pth'
            svm_model_path = r'D:\project\project5001-backend-master\myproject\detection\best_svm_model (2).joblib'
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            rf_model = load_sklearn_model(rf_model_path)
            knn_model = load_sklearn_model(knn_model_path)
            svm_model = load_sklearn_model(svm_model_path)
            dnn_model = load_model(dnn_model_path)
            dnn_model.to(device)

            models = [rf_model, knn_model, svm_model, dnn_model]
            avg_probabilities, predicted_label = soft_voting(models, features)
            print(f"Predicted Label: {predicted_label}")
            print(f"Average Probabilities: {avg_probabilities}")
            labels = ['deception', 'true']
            deception_probability = avg_probabilities[0]
            print(f"Deception Probability: {deception_probability}")

          
          
          
            return Response({
                'Predicted Label': predicted_label,
                'Average Probabilities': avg_probabilities,
                'Deception Probability': deception_probability}, status=status.HTTP_201_CREATED)
        else:
            print(serializer.errors)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
