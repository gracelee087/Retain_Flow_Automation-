import pandas as pd
import streamlit as st
import pickle
from sqlalchemy import create_engine

# 모델 로드
with open("notebook/pipeline_customer_churn_model.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
scaler = bundle["scaler"]
kmeans = bundle["kmeans"]

# 클러스터 라벨 매핑 함수
def label_cluster(cluster):
    if cluster == 2:
        return "High Risk & High Value"
    elif cluster == 0:
        return "Low Risk & High Value"
    elif cluster == 1:
        return "Low Risk & Low Value"
    elif cluster == 3:
        return "Low Risk & Mid Value"
    else:
        return "Unknown"

# DB 연결
engine = create_engine("sqlite:///customers.db")

st.title("고객 이탈 예측 + 세그먼트 데모")

uploaded_file = st.file_uploader("고객 CSV 업로드", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # 예측
    df["churn_prob"] = model.predict_proba(df)[:, 1]

    # 클러스터링
    cluster_input = pd.DataFrame({
        "ChurnProbability": df["churn_prob"],
        "MonthlyCharges": df["MonthlyCharges"]
    })
    df["Cluster"] = kmeans.predict(scaler.transform(cluster_input))

    # 클러스터 해석 라벨
    df["ClusterLabel"] = df["Cluster"].apply(label_cluster)

    # DB 저장
    df.to_sql("predictions", con=engine, if_exists="append", index=False)

    st.write("예측 및 세그먼트 결과:")
    st.dataframe(df[["customerID", "churn_prob", "ClusterLabel"]])

    st.success("DB 저장 완료 → n8n이 자동 메일 발송합니다.")



# ✅ customers.db 핑크 특징

# 파일 기반: .db 파일 하나 = 데이터베이스 전체 (테이블, 인덱스, 데이터 다 포함)

# 서버 불필요: Postgres/MySQL처럼 서버를 켜지 않아도 됨 → Python만 있으면 됨

# SQL 그대로 사용 가능: SELECT, INSERT, UPDATE 같은 SQL 문법 100% 지원

# 이식성 최고: 파일 하나만 복사하면 다른 PC에서도 바로 사용 가능

# ✨ 신기한 점

# Streamlit → Pandas DataFrame → SQLite .db 저장 → VS Code에서 바로 테이블 조회

# 즉, **엑셀 파일처럼 가볍게 쓰는데, SQL DB의 장점(쿼리, 스키마, 확장성)**을 동시에 누릴 수 있음