#import firebase_admin
from flask import Flask, json, request, jsonify
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
# from firebase_admin import credentials, firestore

# cred = credentials.Certificate("covid-c2a93-firebase-adminsdk-kwhe7-3b7ec10615.json")
# firebase_admin.initialize_app(cred)

#db = firestore.client()
# patient = db.collection(u'patients').document(u'patient')
global patient
patient = {}
global patientResult
patientResult = 0.0

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "HI"

# 받은 데이터를 저장
@app.route("/set", methods=["GET", "POST"])
def set():
    global patient
    patient = request.get_json()
    print(logistic())
    return "SET"



# 변수를 불러와 로지스틱 모델에 돌리기
@app.route("/get", methods=["GET", "POST"])
def get():
    print(request.get_json())
    return {
            "id": str(logistic())
    }

def logistic():
    data = pd.read_csv("Data_ver2.csv")  # 데이터 읽기

    features = data[['Age', 'Vascular and unspecified dementia', 'Alzheimer disease', 'Circulatory diseases',
                     'Diabetes', 'Malignant neoplasms', 'Obesity', 'Renal failure', 'Respiratory diseases', 'Sepsis']]
    survival = data['Survived']

    train_features, test_features, train_labels, test_labels = train_test_split(features, survival)  # train, test 분리

    ###### 정규화 ######
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)
    ###################

    model = LogisticRegression()  # 로지스틱 회귀 모델 생성
    model.fit(train_features, train_labels)  # 데이터 fit

    print('5000 Data Trained Model Accuracy: ', round(model.score(train_features, train_labels) * 100, 2), '%',
          sep='')  # 정확도 계산
    print("--------------------------------------------------")

    patientArray = np.array([int(patient["Age"]), int(patient["Vascular and unspecified dementia"]),
                int(patient["Alzheimer disease"]), int(patient["Circulatory diseases"]), int(patient["Diabetes"])
                , int(patient["Malignant neoplasms"]), int(patient["Obesity"]), int(patient["Renal failure"])
                , int(patient["Respiratory diseases"]), int(patient["Sepsis"])])

    patientData = np.array([patientArray])
    patientData = scaler.transform(patientData)  # 데이터 정규화

    print()
    print(model.predict(patientData))  # 예측 결과
    print(model.predict_proba(patientData))  # 예측 확률
    patientResult = model.predict_proba(patientData)[0][0]
    return patientResult        # 사망률 return

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000)
