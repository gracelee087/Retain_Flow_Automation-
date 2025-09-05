import pandas as pd
import streamlit as st
import pickle
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.dialects.postgresql import insert
import numpy as np
import streamlit as st
import os


# 탭 생성
tab1, tab2, tab3, tab4, tab5= st.tabs(["Problem", "EDA", "Modeling/Evaluation", "Application", "Outcome"])



with tab1:
    st.header("Problem - 기본 정보")
    st.write("여기에 다른 기능을 넣을 수 있어요")

with tab2:
    st.header("EDA - 탐색적 데이터 분석 결과")

    eda_path = r"C:\Users\honor\spicedAcademy\Capstone_Final_Project\Retain_Flow_Automation-\notebook\notebook\eda_insight"

    if os.path.exists(eda_path):
        img_files = [f for f in os.listdir(eda_path) if f.endswith((".png", ".jpg", ".jpeg"))]

        if img_files:
            for img in img_files:
                st.image(
                    os.path.join(eda_path, img),
                    caption=img,
                    use_container_width=True  # ✅ 변경됨
                )
        else:
            st.warning("⚠️ EDA 이미지 파일이 없습니다.")
    else:
        st.error("❌ EDA 경로를 찾을 수 없습니다. 경로를 다시 확인해주세요.")



with tab3:
    st.header("Modeling/Evaluation - 모델 결과 확인")
    st.write("샘플 예측 결과, 피처 중요도 등")
    
    st.header("🧪 모델 성능 확인")

    modeling_path = r"C:\Users\honor\spicedAcademy\Capstone_Final_Project\Retain_Flow_Automation-\notebook\notebook\modeling_insight"

    if os.path.exists(modeling_path):
        img_files = [f for f in os.listdir(modeling_path) if f.endswith((".png", ".jpg", ".jpeg"))]

        if img_files:
            for img in sorted(img_files):  # 정렬해서 순서대로 보여주기
                st.image(
                    os.path.join(modeling_path, img),
                    caption=img,
                    width=800 # ✅ 원하는 크기 (px 단위)
                )
        else:
            st.warning("⚠️ 모델 성능 이미지 파일이 없습니다.")
    else:
        st.error("❌ 모델링 결과 경로를 찾을 수 없습니다. 경로를 다시 확인해주세요.")

    # ---------------------------
    # Churn 모델과 Revenue 모델 구분선
    # ---------------------------
    st.divider()  # 최신 Streamlit
    # st.markdown("---")  # 혹은 이 방식도 가능

    st.header("💰 Revenue 모델 성능 시각화")

    revenue_path = r"C:\Users\honor\spicedAcademy\Capstone_Final_Project\Retain_Flow_Automation-\notebook\revenue_insight"

    if os.path.exists(revenue_path):
        img_files = [f for f in os.listdir(revenue_path) if f.endswith(".png")]
        if img_files:
            for img in sorted(img_files):
                st.image(
                    os.path.join(revenue_path, img),
                    caption=img,
                    width=800
                )
        else:
            st.warning("⚠️ Revenue 모델 시각화 이미지가 없습니다.")
    else:
        st.error("❌ Revenue 결과 경로를 찾을 수 없습니다. 경로를 다시 확인해주세요.")







with tab4:
    st.header("Application - 고객 이탈 + 매출 예측 (Supabase 연동)")

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
        "postgresql://postgres:Nwk5JYywxV3ATT8M@db.fjaxvaegmtbsyogavuzy.supabase.co:5432/postgres"
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

        # (2) 매출 예측 (Baseline + Residual)
        X_base = df[["tenure", "MonthlyCharges"]]
        X_res = df[["PaymentMethod", "PaperlessBilling", "Dependents",
                    "OnlineBackup", "InternetService", "StreamingTV", "OnlineSecurity"]]

        baseline_pred = base_model.predict(X_base)
        residual_pred = residual_model.predict(X_res)
        df["predicted_revenue"] = np.clip(baseline_pred + residual_pred, a_min=0, a_max=None)

        # (3) 📌 12개월 기준 지표
        df["revenue_12m"] = df["MonthlyCharges"] * 12
        df["expected_loss_12m"] = df["revenue_12m"] * df["churn_prob"]

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

        # ---------------------------
        # 7. Supabase DB 저장 (전체 고객 → predictions 테이블)
        # ---------------------------
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
                    predicted_revenue=row["predicted_revenue"],
                    revenue_12m=row["revenue_12m"],
                    expected_loss_12m=row["expected_loss_12m"]
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=["customer_id"],
                    set_={
                        "email": row["email"],
                        "churn_prob": row["churn_prob"],
                        "cluster_label": row["cluster_label"],
                        "base_message": row["base_message"],
                        "predicted_revenue": row["predicted_revenue"],
                        "revenue_12m": row["revenue_12m"],
                        "expected_loss_12m": row["expected_loss_12m"]
                    }
                )
                conn.execute(stmt)

        # ---------------------------
        # 8. Top 10 고객 저장 (→ top_risk_customers 테이블)
        # ---------------------------
        top10 = df.sort_values("expected_loss_12m", ascending=False).head(10)

        top_table = metadata.tables["top_risk_customers"]

        with engine.begin() as conn:
            # 기존 데이터 지우고 새로 저장 (덮어쓰기 방식)
            conn.execute(text("TRUNCATE TABLE top_risk_customers;"))

            for _, row in top10.iterrows():
                stmt = insert(top_table).values(
                    customer_id=row["customer_id"],
                    email=row["email"],
                    churn_prob=row["churn_prob"],
                    cluster_label=row["cluster_label"],
                    base_message=row["base_message"],
                    predicted_revenue=row["predicted_revenue"],
                    revenue_12m=row["revenue_12m"],
                    expected_loss_12m=row["expected_loss_12m"]
                )
                conn.execute(stmt)

        # ---------------------------
        # 9. Streamlit 출력
        # ---------------------------
        st.subheader("예측 및 세그먼트 결과")
        st.dataframe(df[["customer_id", "email", "churn_prob",
                        "cluster_label", "base_message",
                        "predicted_revenue",
                        "revenue_12m", "expected_loss_12m"]])

        st.subheader("Top 10 Revenue at Risk 고객 (12M 기준)")
        st.dataframe(top10)

        st.success("✅ Supabase DB 업데이트 완료! (전체 predictions + Top 10 저장)")




with tab5:
    st.header("Outcome - 결론(+product) 및 향후 과제")
    st.write("여기에 다른 기능을 넣을 수 있어요")



















