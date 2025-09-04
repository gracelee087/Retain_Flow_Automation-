import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, MetaData
from sqlalchemy.dialects.postgresql import insert
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# ---------------------------
# 1. 모델 로드 (앱 시작 시 1회 실행)
# ---------------------------
with open("notebook/pipeline_customer_churn_model.pkl", "rb") as f:
    bundle = pickle.load(f)
model = bundle["model"]
scaler = bundle["scaler"]
kmeans = bundle["kmeans"]

with open("notebook/pipeline_customer_revenue_model.pkl", "rb") as f:
    revenue_model = pickle.load(f)

# DB 연결
engine = create_engine(
    "postgresql://postgres:Nwk5JYywxV3ATT8M@db.fjaxvaegmtbsyogavuzy.supabase.co:5432/postgres"
)

# 클러스터 라벨링 함수
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

base_messages = {
    "High Risk & High Value": "프리미엄 고객 전용 혜택 안내",
    "Low Risk & High Value": "VIP 고객님께 드리는 감사 인사",
    "Low Risk & Low Value": "고객님의 소중한 의견을 듣고 싶습니다",
    "Low Risk & Mid Value": "편안한 서비스 이용을 위한 맞춤 제안",
    "Unknown": "기본 안내 메시지"
}

# ---------------------------
# 2. 발표용 탭 구조
# ---------------------------
st.title("📊 고객 이탈 & 매출 예측 프로젝트 - 발표")

tabs = st.tabs(["문제제기", "EDA", "ML 모델링", "비즈니스 적용", "결론 및 향후 과제"])

# ---------------------------
# 3. 문제제기
# ---------------------------
with tabs[0]:
    st.header("문제제기")
    st.write("통신사 고객 이탈률이 높아지고 있습니다. 이탈은 곧 매출 손실로 이어집니다.")

    labels = ["유지 고객", "이탈 고객"]
    sizes = [0.73, 0.27]  # 예시 비율
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')
    st.pyplot(fig)

# ---------------------------
# 4. EDA
# ---------------------------
with tabs[1]:
    st.header("EDA (탐색적 데이터 분석)")
    df = pd.read_csv("customer_Info copy.csv")

    st.subheader("Tenure 분포")
    fig, ax = plt.subplots()
    sns.histplot(df["tenure"], bins=30, ax=ax)
    st.pyplot(fig)

    st.subheader("Contract 유형별 이탈률")
    churn_rate = df.groupby("Contract")["Churn"].apply(lambda x: (x == "Yes").mean())
    st.bar_chart(churn_rate)

# ---------------------------
# 5. ML 모델링
# ---------------------------
with tabs[2]:
    st.header("ML 모델링")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("이탈율 모델 성능")
        st.image("confusion_matrix.png")  # 사전 저장된 이미지
        st.metric("Recall", "0.82")
        st.metric("ROC-AUC", "0.87")

    with col2:
        st.subheader("Revenue 모델 성능")
        st.image("feature_importance.png")  # 사전 저장된 이미지
        st.metric("R²", "0.76")
        st.metric("RMSE", "115.3")

# ---------------------------
# 6. 비즈니스 적용
# ---------------------------
with tabs[3]:
    st.header("비즈니스 적용")

    uploaded_file = st.file_uploader("고객 CSV 업로드", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # (1) 이탈 확률 예측
        df["churn_prob"] = model.predict_proba(df)[:, 1]

        # (2) 매출 예측
        revenue_features = [
            "tenure", "MonthlyCharges", "SeniorCitizen",
            "Contract", "PaymentMethod", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies", "PaperlessBilling",
            "Partner", "Dependents"
        ]
        X_revenue = df[revenue_features]
        df["predicted_revenue"] = np.clip(revenue_model.predict(X_revenue), a_min=0, a_max=None)

        # (3) Expected Loss (12M)
        df["revenue_12m"] = df["MonthlyCharges"] * 12
        df["expected_loss"] = df["revenue_12m"] * df["churn_prob"]

        # (4) 클러스터링
        cluster_input = pd.DataFrame({
            "ChurnProbability": df["churn_prob"],
            "MonthlyCharges": df["MonthlyCharges"]
        })
        df["Cluster"] = kmeans.predict(scaler.transform(cluster_input))
        df["cluster_label"] = df["Cluster"].apply(label_cluster)

        # (5) base_message
        df["base_message"] = df["cluster_label"].map(base_messages)

        # (6) DB 저장
        metadata = MetaData()
        metadata.reflect(bind=engine)
        predictions_table = metadata.tables["predictions"]

        with engine.begin() as conn:
            for _, row in df.iterrows():
                stmt = insert(predictions_table).values(
                    customer_id=row.get("customerID"),
                    email=row.get("Email"),
                    churn_prob=row["churn_prob"],
                    cluster_label=row["cluster_label"],
                    base_message=row["base_message"],
                    predicted_revenue=row["predicted_revenue"],
                    expected_loss=row["expected_loss"],
                    revenue_12m=row["revenue_12m"]
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=["customer_id"],
                    set_={
                        "email": row.get("Email"),
                        "churn_prob": row["churn_prob"],
                        "cluster_label": row["cluster_label"],
                        "base_message": row["base_message"],
                        "predicted_revenue": row["predicted_revenue"],
                        "expected_loss": row["expected_loss"],
                        "revenue_12m": row["revenue_12m"]
                    }
                )
                conn.execute(stmt)

        # (7) 발표용 출력
        st.subheader("예측 및 세그먼트 결과")
        st.dataframe(df[[
            "customerID", "Email", "churn_prob",
            "cluster_label", "base_message",
            "predicted_revenue", "expected_loss", "revenue_12m"
        ]])

        st.subheader("Top 10 Revenue at Risk 고객 (12M 기준)")
        st.dataframe(df.sort_values("expected_loss", ascending=False).head(10))

        # (8) 클러스터링 시각화
        st.subheader("고객 세그먼트 시각화")
        fig = px.scatter(
            df,
            x="MonthlyCharges", y="churn_prob",
            color="cluster_label",
            hover_data=["customerID", "predicted_revenue", "expected_loss"],
            title="Risk vs Value 세그먼트"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.success("✅ Supabase DB에 최신 정보 저장 완료! (중복 고객은 업데이트됨)")

# ---------------------------
# 7. 결론 및 향후 과제
# ---------------------------
with tabs[4]:
    st.header("결론 및 향후 과제")
    st.success("이탈율 모델 + Revenue 모델을 활용해 고객 유지 전략 수립 가능")

    st.markdown("""
    **향후 과제**
    - Survival Analysis로 LTV 정밀화  
    - 실시간 데이터 파이프라인 구축  
    - 캠페인 자동화 (혜택 제공)  
    """)
