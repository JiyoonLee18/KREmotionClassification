import os
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from keras.models import load_model
from sentence_transformers import SentenceTransformer

from utils import get_features, text_embedding_api

# 모델 경로 설정
current_path=os.getcwd().replace("\\", "/").replace("c:", "C:")
model_path = current_path+'/model/'  # 모델 파일이 위치한 경로를 설정해야 합니다.

# 모델 로드
model = load_model(model_path+'my_model.h5')
encoder = joblib.load(model_path+'encoder.pkl')
scaler = joblib.load(model_path+'scaler.pkl')
with open(model_path+'text_model_name.txt', 'r') as file:
    text_model_name = file.read().strip()

# 입력 데이터 클래스
class InputData(BaseModel):
    audioFilePath: str
    sentence: str

app = FastAPI()

@app.post("/predict/")
async def make_prediction(data: InputData):
    try:
        new_audio_features = get_features(data.audioFilePath)
        new_audio_features_df = pd.DataFrame([new_audio_features])
        new_audio_features_df['sentence'] = data.sentence
    
        txt_embed = text_embedding_api(model_name=text_model_name)
        new_audio_features_df = txt_embed.transform(new_audio_features_df)

        # 특징 표준화
        new_audio_features_scaled = scaler.transform(new_audio_features_df)

        # 모델에 맞게 차원 확장
        new_audio_features_scaled = np.expand_dims(new_audio_features_scaled, axis=2)

        # 예측 수행
        prediction = model.predict(new_audio_features_scaled)
        predicted_label = np.argmax(prediction, axis=1)
        num_categories = encoder.categories_[0].size

        # 모델의 예측 결과를 원-핫 인코딩 형식으로 변환
        predicted_one_hot = np.zeros((predicted_label.size, num_categories))
        predicted_one_hot[np.arange(predicted_label.size), predicted_label] = 1

        # 원-핫 인코딩된 예측 결과를 실제 레이블로 변환
        actual_label = encoder.inverse_transform(predicted_one_hot)

        return {
            "label": actual_label[0][0],
            "predictions": prediction.tolist()  # 확률 분포 반환
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# FastAPI 서버 실행 (예시: uvicorn main:app --reload)