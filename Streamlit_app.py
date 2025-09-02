import pandas as pd
import streamlit as st
import pickle
from sqlalchemy import create_engine, MetaData
from sqlalchemy.dialects.postgresql import insert
import numpy as np
from dateutil.relativedelta import relativedelta

# ---------------------------
# 1. 모델 로드
# ---------------------------
with open("notebook/pipeline_customer_churn_model.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
scaler = bundle["scaler"]
kmeans = bundle["kmeans"]

with open("notebook/pipeline_customer_revenue_model.pkl", "rb") as f:
    revenue_bundle = pickle.load(f)

base_model = revenue_bundle["baseline_model"]
residual_model = revenue_bundle["residual_model"]

# ---------------------------
# 2. Postgres DB 연결
# ---------------------------
engine = create_engine(
    "postgresql://postgres:PqKHbS8fqXKSnyYv@db.fjaxvaegmtbsyogavuzy.supabase.co:5432/postgres"
)

# ---------------------------
# 3. 세그먼트 라벨링 함수 & base_message 매핑
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

base_messages = {
    "High Risk & High Value": "프리미엄 고객 전용 혜택 안내",
    "Low Risk & High Value": "VIP 고객님께 드리는 감사 인사",
    "Low Risk & Low Value": "고객님의 소중한 의견을 듣고 싶습니다",
    "Low Risk & Mid Value": "편안한 서비스 이용을 위한 맞춤 제안",
    "Unknown": "기본 안내 메시지"
}

# ---------------------------
# 4. Streamlit UI
# ---------------------------
st.title("📊 고객 이탈 + 매출 예측 (Supabase 연동)")

uploaded_file = st.file_uploader("고객 CSV 업로드", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # (1) 고객 이탈 확률 예측
    df["churn_prob"] = model.predict_proba(df)[:, 1]

    # (2) 매출 예측 (Baseline + Residual → LTV 기반)
    X_base = df[["tenure", "MonthlyCharges"]]
    X_res = df[["PaymentMethod", "PaperlessBilling", "Dependents",
                "OnlineBackup", "InternetService", "StreamingTV", "OnlineSecurity"]]

    baseline_pred = base_model.predict(X_base)
    residual_pred = residual_model.predict(X_res)
    df["predicted_revenue"] = np.clip(baseline_pred + residual_pred, a_min=0, a_max=None)

    # (3) Revenue at Risk (Expected Loss, LTV 기준)
    df["expected_loss"] = df["churn_prob"] * df["predicted_revenue"]

    # (3-b) 📌 12개월 기준 지표 추가 (Telco 표준)
    df["revenue_12m"] = df["MonthlyCharges"] * 12
    df["expected_loss_12m"] = df["revenue_12m"] * df["churn_prob"]

    # (3-c) LTV 근사치 (간단히 1/churn_prob 개월 남는다고 가정)
    df["expected_months_remaining"] = df["churn_prob"].apply(lambda p: 1/p if p > 0 else 60)  # 최대 60개월 cap
    df["ltv"] = df["MonthlyCharges"] * df["expected_months_remaining"]
    df["expected_loss_ltv"] = df["ltv"] * df["churn_prob"]

    # (3-d) 가입일(start_date) 추정
    today = pd.to_datetime("2025-09-01")  # 기준일
    df["start_date"] = df["tenure"].apply(lambda m: today - relativedelta(months=int(m)))

    # (4) 클러스터링
    cluster_input = pd.DataFrame({
        "ChurnProbability": df["churn_prob"],
        "MonthlyCharges": df["MonthlyCharges"]
    })
    df["Cluster"] = kmeans.predict(scaler.transform(cluster_input))
    df["cluster_label"] = df["Cluster"].apply(label_cluster)

    # (5) base_message 생성
    df["base_message"] = df["cluster_label"].map(base_messages)

    # (6) 컬럼명 DB 테이블과 맞추기
    df = df.rename(columns={
        "customerID": "customer_id",
        "Email": "email"
    })

    # (7) Supabase DB 저장 (UPSERT)
    metadata = MetaData()
    metadata.reflect(bind=engine)
    predictions_table = metadata.tables["predictions"]

    with engine.begin() as conn:
        for _, row in df.iterrows():
            stmt = insert(predictions_table).values(
                customer_id=row["customer_id"],
                email=row["email"],
                churn_prob=row["churn_prob"],
                cluster_label=row["cluster_label"],
                base_message=row["base_message"],
                predicted_revenue=row["predicted_revenue"],     # LTV 기반
                expected_loss=row["expected_loss"],             # LTV 기반
                revenue_12m=row["revenue_12m"],                 # 12M 기준
                expected_loss_12m=row["expected_loss_12m"],     # 12M 기준
                ltv=row["ltv"],                                 # 생애 가치
                expected_loss_ltv=row["expected_loss_ltv"],     # LTV 손실 위험
                start_date=row["start_date"]
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["customer_id"],
                set_={
                    "email": row["email"],
                    "churn_prob": row["churn_prob"],
                    "cluster_label": row["cluster_label"],
                    "base_message": row["base_message"],
                    "predicted_revenue": row["predicted_revenue"],
                    "expected_loss": row["expected_loss"],
                    "revenue_12m": row["revenue_12m"],
                    "expected_loss_12m": row["expected_loss_12m"],
                    "ltv": row["ltv"],
                    "expected_loss_ltv": row["expected_loss_ltv"],
                    "start_date": row["start_date"]
                }
            )
            conn.execute(stmt)

    # (8) Streamlit 출력
    st.subheader("예측 및 세그먼트 결과")
    st.dataframe(df[["customer_id", "email", "churn_prob",
                     "cluster_label", "base_message",
                     "predicted_revenue", "expected_loss",
                     "revenue_12m", "expected_loss_12m",
                     "ltv", "expected_loss_ltv",
                     "start_date"]])

    st.subheader("Top 10 Revenue at Risk 고객 (LTV 기준)")
    st.dataframe(df.sort_values("expected_loss", ascending=False).head(10))

    st.subheader("Top 10 Revenue at Risk 고객 (12M 기준)")
    st.dataframe(df.sort_values("expected_loss_12m", ascending=False).head(10))

    st.success("✅ Supabase DB에 최신 정보 저장 완료! (중복 고객은 업데이트됨)")































# 2025 09 02 - (2) 
# import pandas as pd
# import streamlit as st
# import pickle
# from sqlalchemy import create_engine, MetaData
# from sqlalchemy.dialects.postgresql import insert
# import numpy as np   # ✅ 추가



# # ---------------------------
# # 1. 모델 로드
# # ---------------------------
# # (1) 고객 이탈 예측 모델
# with open("notebook/pipeline_customer_churn_model.pkl", "rb") as f:
#     bundle = pickle.load(f)

# model = bundle["model"]
# scaler = bundle["scaler"]
# kmeans = bundle["kmeans"]

# # (2) 고객 매출 예측 모델
# with open("notebook/pipeline_customer_revenue_model.pkl", "rb") as f:
#     revenue_bundle = pickle.load(f)

# base_model = revenue_bundle["baseline_model"]
# residual_model = revenue_bundle["residual_model"]

# # ---------------------------
# # 2. Postgres DB 연결
# # ---------------------------
# engine = create_engine(
#     "postgresql://postgres:PqKHbS8fqXKSnyYv@db.fjaxvaegmtbsyogavuzy.supabase.co:5432/postgres"
# )

# # ---------------------------
# # 3. 세그먼트 라벨링 함수 & base_message 매핑
# # ---------------------------
# def label_cluster(cluster):
#     if cluster == 2:
#         return "High Risk & High Value"
#     elif cluster == 0:
#         return "Low Risk & High Value"
#     elif cluster == 1:
#         return "Low Risk & Low Value"
#     elif cluster == 3:
#         return "Low Risk & Mid Value"
#     else:
#         return "Unknown"

# base_messages = {
#     "High Risk & High Value": "프리미엄 고객 전용 혜택 안내",
#     "Low Risk & High Value": "VIP 고객님께 드리는 감사 인사",
#     "Low Risk & Low Value": "고객님의 소중한 의견을 듣고 싶습니다",
#     "Low Risk & Mid Value": "편안한 서비스 이용을 위한 맞춤 제안",
#     "Unknown": "기본 안내 메시지"
# }

# # ---------------------------
# # 4. Streamlit UI
# # ---------------------------
# st.title("📊 고객 이탈 + 매출 예측 (Supabase 연동)")

# uploaded_file = st.file_uploader("고객 CSV 업로드", type="csv")

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)

#     # (1) 고객 이탈 확률 예측
#     df["churn_prob"] = model.predict_proba(df)[:, 1]

#     # (2) 매출 예측 (Baseline + Residual)
#     X_base = df[["tenure", "MonthlyCharges"]]
#     X_res = df[["PaymentMethod", "PaperlessBilling", "Dependents",
#                 "OnlineBackup", "InternetService", "StreamingTV", "OnlineSecurity"]]

#     baseline_pred = base_model.predict(X_base)
#     residual_pred = residual_model.predict(X_res)



# # ✅ 매출은 음수가 될 수 없으므로 NumPy clip 사용
#     df["predicted_revenue"] = np.clip(baseline_pred + residual_pred, a_min=0, a_max=None)

# # (3) Revenue at Risk (Expected Loss)
#     df["expected_loss"] = df["churn_prob"] * df["predicted_revenue"]



# # ✅ 월 단위 지표 추가
#     df["monthly_predicted_revenue"] = df.apply(
#     lambda row: row["predicted_revenue"] / row["tenure"] if row["tenure"] > 0 else 0,
#     axis=1
#     )

#     df["monthly_expected_loss"] = df.apply(
#     lambda row: row["expected_loss"] / row["tenure"] if row["tenure"] > 0 else 0,
#     axis=1
#     )




#     # (4) 클러스터링
#     cluster_input = pd.DataFrame({
#         "ChurnProbability": df["churn_prob"],
#         "MonthlyCharges": df["MonthlyCharges"]
#     })
#     df["Cluster"] = kmeans.predict(scaler.transform(cluster_input))
#     df["cluster_label"] = df["Cluster"].apply(label_cluster)

#     # (5) base_message 생성
#     df["base_message"] = df["cluster_label"].map(base_messages)

#     # (6) 컬럼명 DB 테이블과 맞추기
#     df = df.rename(columns={
#         "customerID": "customer_id",
#         "Email": "email"
#     })

#     # (7) Supabase DB 저장 (UPSERT)
#     metadata = MetaData()
#     metadata.reflect(bind=engine)
#     predictions_table = metadata.tables["predictions"]

#     with engine.begin() as conn:
#         for _, row in df.iterrows():
#             stmt = insert(predictions_table).values(
#                 customer_id=row["customer_id"],
#                 email=row["email"],
#                 churn_prob=row["churn_prob"],
#                 cluster_label=row["cluster_label"],
#                 base_message=row["base_message"],
#                 predicted_revenue=row["predicted_revenue"],
#                 expected_loss=row["expected_loss"]
#             )
#             # ✅ customer_id가 이미 있으면 업데이트
#             stmt = stmt.on_conflict_do_update(
#                 index_elements=["customer_id"],
#                 set_={
#                     "email": row["email"],
#                     "churn_prob": row["churn_prob"],
#                     "cluster_label": row["cluster_label"],
#                     "base_message": row["base_message"],
#                     "predicted_revenue": row["predicted_revenue"],
#                     "expected_loss": row["expected_loss"]
#                 }
#             )
#             conn.execute(stmt)

#     # (8) Streamlit 출력
#     st.subheader("예측 및 세그먼트 결과")
#     st.dataframe(df[["customer_id", "email", "churn_prob",
#                      "cluster_label", "base_message",
#                      "predicted_revenue", "expected_loss"]])

#     st.subheader("Top 10 Revenue at Risk 고객")
#     st.dataframe(df.sort_values("expected_loss", ascending=False).head(10))

#     st.success("✅ Supabase DB에 최신 정보 저장 완료! (중복 고객은 업데이트됨)")

























# 2025 / 09 / 02 - (1)
# import pandas as pd
# import streamlit as st
# import pickle
# from sqlalchemy import create_engine, MetaData
# from sqlalchemy.dialects.postgresql import insert

# # ---------------------------
# # 1. 모델 로드
# # ---------------------------
# with open("notebook/pipeline_customer_churn_model.pkl", "rb") as f:
#     bundle = pickle.load(f)

# model = bundle["model"]
# scaler = bundle["scaler"]
# kmeans = bundle["kmeans"]

# # ---------------------------
# # 2. Postgres DB 연결
# # ---------------------------
# engine = create_engine(
#     "postgresql://postgres:PqKHbS8fqXKSnyYv@db.fjaxvaegmtbsyogavuzy.supabase.co:5432/postgres"
# )

# # ---------------------------
# # 3. 세그먼트 라벨링 함수 & base_message 매핑
# # ---------------------------
# def label_cluster(cluster):
#     if cluster == 2:
#         return "High Risk & High Value"
#     elif cluster == 0:
#         return "Low Risk & High Value"
#     elif cluster == 1:
#         return "Low Risk & Low Value"
#     elif cluster == 3:
#         return "Low Risk & Mid Value"
#     else:
#         return "Unknown"

# base_messages = {
#     "High Risk & High Value": "프리미엄 고객 전용 혜택 안내",
#     "Low Risk & High Value": "VIP 고객님께 드리는 감사 인사",
#     "Low Risk & Low Value": "고객님의 소중한 의견을 듣고 싶습니다",
#     "Low Risk & Mid Value": "편안한 서비스 이용을 위한 맞춤 제안",
#     "Unknown": "기본 안내 메시지"
# }

# # ---------------------------
# # 4. Streamlit UI
# # ---------------------------
# st.title("고객 이탈 예측 + 세그먼트 데모 (Supabase 버전)")

# uploaded_file = st.file_uploader("고객 CSV 업로드", type="csv")

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)

#     # (1) 고객 이탈 확률 예측
#     df["churn_prob"] = model.predict_proba(df)[:, 1]

#     # (2) 클러스터링
#     cluster_input = pd.DataFrame({
#         "ChurnProbability": df["churn_prob"],
#         "MonthlyCharges": df["MonthlyCharges"]
#     })
#     df["Cluster"] = kmeans.predict(scaler.transform(cluster_input))
#     df["cluster_label"] = df["Cluster"].apply(label_cluster)

#     # (3) base_message 생성
#     df["base_message"] = df["cluster_label"].map(base_messages)

#     # (4) 컬럼명 DB 테이블과 맞추기
#     df = df.rename(columns={
#         "customerID": "customer_id",
#         "Email": "email"
#     })

#     # (5) Supabase DB 저장 (UPSERT)
#     metadata = MetaData()
#     metadata.reflect(bind=engine)
#     predictions_table = metadata.tables["predictions"]

#     with engine.begin() as conn:
#         for _, row in df.iterrows():
#             stmt = insert(predictions_table).values(
#                 customer_id=row["customer_id"],
#                 email=row["email"],
#                 churn_prob=row["churn_prob"],
#                 cluster_label=row["cluster_label"],
#                 base_message=row["base_message"]
#             )
#             # ✅ customer_id가 이미 있으면 업데이트
#             stmt = stmt.on_conflict_do_update(
#                 index_elements=["customer_id"],
#                 set_={
#                     "email": row["email"],
#                     "churn_prob": row["churn_prob"],
#                     "cluster_label": row["cluster_label"],
#                     "base_message": row["base_message"]
#                 }
#             )
#             conn.execute(stmt)

#     # (6) Streamlit 출력
#     st.subheader("예측 및 세그먼트 결과")
#     st.dataframe(df[["customer_id", "email", "churn_prob", "cluster_label", "base_message"]])

#     st.success("✅ Supabase DB에 최신 정보 저장 완료! (중복 고객은 업데이트됨)")



































# ✅ customers.db 핑크 특징

# 파일 기반: .db 파일 하나 = 데이터베이스 전체 (테이블, 인덱스, 데이터 다 포함)

# 서버 불필요: Postgres/MySQL처럼 서버를 켜지 않아도 됨 → Python만 있으면 됨

# SQL 그대로 사용 가능: SELECT, INSERT, UPDATE 같은 SQL 문법 100% 지원

# 이식성 최고: 파일 하나만 복사하면 다른 PC에서도 바로 사용 가능

# ✨ 신기한 점

# Streamlit → Pandas DataFrame → SQLite .db 저장 → VS Code에서 바로 테이블 조회

# 즉, **엑셀 파일처럼 가볍게 쓰는데, SQL DB의 장점(쿼리, 스키마, 확장성)**을 동시에 누릴 수 있음