import pandas as pd
import streamlit as st
import pickle
from sqlalchemy import create_engine

# ---------------------------
# 1. 모델 로드
# ---------------------------
with open("notebook/pipeline_customer_churn_model.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
scaler = bundle["scaler"]
kmeans = bundle["kmeans"]

# ---------------------------
# 2. Postgres DB 연결
# ---------------------------
engine = create_engine("postgresql://postgres:PqKHbS8fqXKSnyYv@db.fjaxvaegmtbsyogavuzy.supabase.co:5432/postgres")

# ---------------------------
# 3. 클러스터 라벨링 함수
# ---------------------------
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

# ---------------------------
# 4. Streamlit UI
# ---------------------------
st.title("고객 이탈 예측 + 세그먼트 데모 (Supabase 버전)")

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
    df["ClusterLabel"] = df["Cluster"].apply(label_cluster)

    # 컬럼명 Supabase 테이블에 맞추기
    df = df.rename(columns={
        "customerID": "customer_id",
        "ClusterLabel": "cluster_label",
        "Email": "email"
    })

    # Supabase 저장
    df[["customer_id", "email", "churn_prob", "cluster_label"]].to_sql(
        "predictions", con=engine, if_exists="append", index=False
    )

    st.subheader("예측 및 세그먼트 결과")
    st.dataframe(df[["customer_id", "email", "churn_prob", "cluster_label"]])

    st.success("✅ Supabase DB에 저장 완료!")







    # # 예측 결과 미리보기
    # st.dataframe(df.head(10))

    # # --------------------------
    # # 1) 세그먼트 분포 시각화
    # # --------------------------
    # import matplotlib.pyplot as plt
    # st.subheader("세그먼트 분포")
    # seg_counts = df["ClusterLabel"].value_counts()
    # fig, ax = plt.subplots()
    # ax.pie(seg_counts, labels=seg_counts.index, autopct='%1.1f%%')
    # st.pyplot(fig)

    # # --------------------------
    # # 2) Top High Risk 고객 미리보기
    # # --------------------------
    # st.subheader("Top High Risk 고객")
    # st.dataframe(df.sort_values("churn_prob", ascending=False).head(5))

    # # --------------------------
    # # 3) Retention Action 추천
    # # --------------------------
    # def recommend_action(label):
    #     if label == "High Risk & High Value":
    #         return "VIP 쿠폰 발송"
    #     elif label == "High Risk & Low Value":
    #         return "저비용 뉴스레터"
    #     else:
    #         return "감사 메일"

    # df["Action"] = df["ClusterLabel"].apply(recommend_action)
    # st.subheader("고객별 추천 액션")
    # st.dataframe(df[["customerID", "churn_prob", "ClusterLabel", "Action"]])



# ✅ customers.db 핑크 특징

# 파일 기반: .db 파일 하나 = 데이터베이스 전체 (테이블, 인덱스, 데이터 다 포함)

# 서버 불필요: Postgres/MySQL처럼 서버를 켜지 않아도 됨 → Python만 있으면 됨

# SQL 그대로 사용 가능: SELECT, INSERT, UPDATE 같은 SQL 문법 100% 지원

# 이식성 최고: 파일 하나만 복사하면 다른 PC에서도 바로 사용 가능

# ✨ 신기한 점

# Streamlit → Pandas DataFrame → SQLite .db 저장 → VS Code에서 바로 테이블 조회

# 즉, **엑셀 파일처럼 가볍게 쓰는데, SQL DB의 장점(쿼리, 스키마, 확장성)**을 동시에 누릴 수 있음