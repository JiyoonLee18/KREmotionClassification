# KREmotionClassification
- Korean Emotion Classification using audio, text
- using data 감정 분류를 위한 대화 음성 데이터셋 ai hub [link](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=263)
- code reference [link](https://velog.io/@bandi12/%ED%85%8D%EC%8A%A4%ED%8A%B8%EC%99%80-%EC%9D%8C%EC%84%B1-%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%9C-%ED%95%9C%EA%B5%AD%EC%96%B4-%EA%B0%90%EC%A0%95-%EB%B6%84%EB%A5%98-%EB%AA%A8%EB%8D%B8-2) 
- 파일 구성
  - data 폴더
    - 5차_wav: ai hub 데이터 다운 저장 위치
    - 5차_10011.csv: wav_id따른 텍스트 정보
    - final_df.csv: sampling한 데이터 오디오 feature를 저장한 데이터
  - model 폴더
    - encoder.pkl: 인코더 모델
    - my_model.h5: 감정 분류 모델
    - scaler.pkl: 스케일러 모델
    - text_model_name.txt: 사용한 sbert 모델
  - 1\. preprocessing.ipynb: wav파일 샘플링 및 feature 추출 및 저장
  - 2\. modeling.ipynb: tensorflow==2.10.1 로 모델 학습 및 저장
    - 학습 환경
        - python==3.10
        - tensorflow==2.10.1
        - keras==2.10.0
  - 3\. predict.ipynb: cpu환경에서 모델 불러오기 및 새로운 데이터 예측 예시 노트북 파일
    - 예측 환경
        - python==3.10
        - tensorflow-cpu==2.10.1
        - keras==2.10.0
  - utils.py: 함수 저장 파일


```mermaid
  graph TD;
      A-->B;
      A-->C;
      B-->D;
      C-->D;
```
