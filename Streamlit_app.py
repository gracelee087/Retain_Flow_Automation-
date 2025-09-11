import pandas as pd
import streamlit as st
import pickle
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.dialects.postgresql import insert
import numpy as np
import streamlit as st
import os

from sqlalchemy import create_engine

# Supabase DB 접속 문자열 (비밀번호는 실제 값으로 교체!)
DATABASE_URL = "postgresql+psycopg2://postgres:<비밀번호>@db.fjaxvaegmtbsyogavuzy.supabase.co:5432/postgres?sslmode=require"

try:
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        result = conn.execute("SELECT version();")
        for row in result:
            print("✅ 연결 성공:", row[0])
except Exception as e:
    print("❌ 연결 실패:", e)


    
st.title("RetainFlow Automation: Customer Churn Prediction")


# 탭 생성
tab1, tab2, tab3, tab4, tab5= st.tabs(["Problem", "EDA", "Modeling/Evaluation", "Application", "Outcome"])



with tab1:

    # Objective
    st.header("Objective")
    st.markdown("""
    - Identify churn-prone customers using historical telco data  
    - Automate retention workflows by connecting predictions to **make.com**  
    - Enable timely customer engagement (e.g., sending offers, alerts to CRM, or triggering support tickets)  
    """)


    st.header("Business Problem")
    st.markdown("""
    - Telecom customers can easily switch, so **churn directly reduces revenue**.  
    - Even a small improvement matters: a **5% reduction in churn can boost profits by 25–95%** (Bain & Co.).  
    - **Customer Acquisition Cost (CAC)** is about **5x higher** than **Customer Retention Cost (CRC)** (HBR).  
    - Despite this, many churn analyses stop at *prediction*, without seamless integration into **real business workflows**.  
    - Marketing and support teams often lack **real-time triggers** to act on churn insights.  
    - Without automation, valuable time is lost between churn detection and customer outreach.  
    - **Key challenge**: How well churn is predicted **and** how effectively marketing actions are automated determines revenue growth.  
    - This project bridges the gap by combining **machine learning models** with **make.com automation**, enabling organizations to not only *predict churn* but also to **automatically take retention actions**.  
    """)



    st.header("Telco Opportunity Map")
    st.image("pic.png", caption="Opportunity Map: Balancing Effort vs. Benefit", width=700)

    st.markdown("""
This opportunity map helps us **prioritize projects**:  
- **Top-left (Quick Wins)**: High benefit, low effort → e.g., Churn AI + Discounts, Loyalty programs.  
- **Bottom-right (Long-term Bets)**: High effort, uncertain benefit → e.g., IoT, 5G marketing.  
- Our focus starts with **Churn AI**, where benefit is high and execution is feasible.  
    """)



    st.header("Why It Matters")
    st.markdown("""
- **Cost efficiency**: Retaining is cheaper & more profitable.  
- **Customer lifetime value**: Identify & prioritize high-value churn-risk customers.  
- **Personalized marketing**: Segmentation + churn probability → tailored offers.  
- **Revenue impact**: Proactive churn management drives growth.  
- **Strategic decisions**: Data-driven CRM, bundles, loyalty programs.  
    """)




    # 참고문헌 (작게 표시)
    st.markdown(
        """
        <sub>**References**:  
        Bain & Company, *Customer Retention Economics*  
        Harvard Business Review (2014), *The Value of Keeping the Right Customers*</sub>
        """,
        unsafe_allow_html=True
    )














with tab2:


    
    st.header("Exploratory Data Analysis Results")




    st.markdown("""
                

    ### Data source / collection / challenges

    **Data source**  
    - Telco Customer Churn dataset (Kaggle / IBM Sample Data)  
    - Includes customer contracts, payment methods, service usage, billing history, and churn labels  

    **Data collection**  
    - Mainly gathered from **Customer Relationship Management (CRM)** and **Billing systems**  
    - Key feature categories:  
    - **Customer info**: tenure, SeniorCitizen  
    - **Contract info**: Contract, PaymentMethod, InternetService  
    - **Service usage**: TechSupport, OnlineSecurity, StreamingTV  
    - **Billing data**: MonthlyCharges, TotalCharges  
    - Churn is defined as whether a customer discontinued the service within a given period  

    **Challenges**  
    - **Data quality issues**: missing or invalid `TotalCharges` values → preprocessing required  
    - **Class imbalance**: majority of customers are `Churn=No`, while `Churn=Yes` is a minority → risk of biased models  
    - **Categorical features**: contract, payment method, and service usage require encoding for ML models  
    - **Limitations in realism**:  
    - Lacks behavioral data such as complaints, service quality issues, or customer interactions  
    - No information on customer re-subscription after churn or the impact of specific marketing campaigns  
    - Therefore, assessing marketing effectiveness and retention strategies is challenging with this dataset

    ---

                
    ### Key Insights from EDA

    **Customer Tenure**  
    - Customers with shorter tenure show a higher churn rate  
    - Long-term customers tend to have higher TotalCharges and lower churn probability  

    **Billing Metrics**  
    - Higher MonthlyCharges are associated with higher churn  
    - In contrast, TotalCharges show a negative correlation with churn, reflecting stronger customer loyalty  

    **Contract & Service Features**  
    - Month-to-month contracts have the highest churn rate  
    - Customers paying via **Electronic check** are more likely to churn  
    - Customers without **TechSupport / OnlineSecurity** services show significantly higher churn  

    """)











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




    st.header("Churn Methodology and Technology Stack Used")

    methodology_data = [
        ["Problem Definition", 
         "Predict customer churn in advance and identify high-value customers → design targeted retention strategies"],
        ["Data Preparation", 
         "- Source: customer_Info copy.csv\n- Convert TotalCharges to numeric & handle missing values\n- Transform Churn: Yes/No → 1/0"],
        ["Preprocessing Strategy", 
         "- Numerical: tenure, MonthlyCharges, TotalCharges\n- Categorical: Contract type, Payment method, etc. → OneHotEncoding\n- Numeric cleaning with FunctionTransformer"],
        ["Imbalance Handling", 
         "SMOTE: oversampling of minority class (churned customers)"],
        ["Process", 
         "Model training → Hyperparameter tuning → Probability calibration → Customer segmentation"],
        ["Technology Stack", 
         "- Python (pandas, numpy, scikit-learn, imbalanced-learn, seaborn, matplotlib)\n"
         "- Modeling: Pipeline, OneHotEncoder, ColumnTransformer, FunctionTransformer\n"
         "- Model: RandomForestClassifier + CalibratedClassifierCV\n"
         "- Optimization: GridSearchCV + StratifiedKFold (scoring=recall)\n"
         "- Imbalance: SMOTE\n"
         "- Clustering: KMeans + StandardScaler\n"
         "- Deployment: cloudpickle (model + scaler + clusterer bundle)"]
    ]
    methodology_df = pd.DataFrame(methodology_data, columns=["Section", "Details"])
    st.table(methodology_df)   # ✅ 자동 줄바꿈 표

    st.header("Modelling")

    modelling_data = [
        ["Base Model", "RandomForestClassifier (class_weight='balanced')"],
        ["Hyperparameter Tuning", 
         "GridSearchCV (n_estimators, max_depth, min_samples_split, max_features)\n"
         "5-Fold Stratified CV, optimized for Recall"],
        ["Why Recall", 
         "Missing churners is more costly for the business than false positives"],
        ["Performance Improvements", 
         "- CalibratedClassifierCV: probability calibration (sigmoid)\n"
         "- Threshold adjustment: 0.5 vs 0.3 → improves Recall\n"
         "- ROC Curve: AUC ≈ 0.85"],
        ["Feature Importance", 
         "- Top features: MonthlyCharges, tenure, Contract, InternetService, OnlineSecurity …\n"
         "- Insights:\n"
         "  • High MonthlyCharges + short tenure → higher churn risk\n"
         "  • Missing add-ons (TechSupport, OnlineSecurity) → higher churn risk"],
        ["Segmentation (KMeans)", 
         "- Input: Predicted churn probability + MonthlyCharges\n"
         "- Result: 4 clusters\n"
         "  • Cluster 2: High Risk & High Value (priority customers)\n"
         "  • Cluster 0/1: Low Risk groups\n"
         "- Use case: targeted marketing strategies per segment"]
    ]
    modelling_df = pd.DataFrame(modelling_data, columns=["Section", "Details"])
    st.table(modelling_df)   # ✅ 자동 줄바꿈 표
























    st.header("Modeling/Evaluation")
    
    st.header("Churn Prediction Model Performance Evaluation")

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
    st.header("Revenue Model - Methodology and Technology Stack Used")

    revenue_methodology_data = [
        ["Problem Definition", 
         "Predict customer lifetime revenue (TotalCharges) more accurately by combining baseline trends with advanced modeling"],
        ["Data Preparation", 
         "- Source: customer_Info copy.csv\n- Drop customerID\n- Convert TotalCharges to numeric & fill missing values"],
        ["Preprocessing Strategy", 
         "- Baseline: tenure, MonthlyCharges → Linear Regression\n- Residual: categorical features (Contract, PaymentMethod, InternetService, add-ons like TechSupport, OnlineSecurity) → OneHotEncoding"],
        ["Residual Concept", 
         "Residual = Actual TotalCharges – Baseline prediction\nRandomForestRegressor predicts these residuals"],
        ["Process", 
         "1. Baseline model (Linear Regression)\n2. Compute residuals\n3. Train RandomForestRegressor on residuals\n4. Final prediction = Baseline + Residual model"],
        ["Technology Stack", 
         "- Python (pandas, numpy, scikit-learn, seaborn, matplotlib)\n"
         "- Models: LinearRegression + RandomForestRegressor\n"
         "- Pipeline + ColumnTransformer + OneHotEncoder\n"
         "- Metrics: R², RMSE\n"
         "- Visualization: scatter plots, residual histograms, feature importances"]
    ]
    revenue_methodology_df = pd.DataFrame(revenue_methodology_data, columns=["Section", "Details"])
    st.table(revenue_methodology_df)

    st.header("Revenue Model - Modelling")

    revenue_modelling_data = [
        ["Baseline Model", "Linear Regression with tenure × MonthlyCharges"],
        ["Baseline Performance", "R² ≈ 0.89 → explains ~89% of revenue variance"],
        ["Residual Modeling", "RandomForestRegressor trained on categorical features (Contract, PaymentMethod, TechSupport, etc.)"],
        ["Residual Performance", "R² ≈ 0.55 → explains ~55% of variance in residuals"],
        ["Final Model", "Final prediction = Baseline + Residual model"],
        ["Final Performance", "R² ≈ 0.965, RMSE ≈ 424\nAverage revenue ≈ 2280 → error ≈ 18.6%"],
        ["Feature Importance (Residual Model)", "Key drivers: Contract type, InternetService, PaymentMethod, TechSupport, OnlineSecurity"],
        ["Visualization", "- Baseline vs Actual (scatter)\n- Residual distribution (histogram)\n- Residual Feature Importances (barplot)\n- Final Actual vs Predicted (scatter)\n- Final Residuals (histogram)"]
    ]
    revenue_modelling_df = pd.DataFrame(revenue_modelling_data, columns=["Section", "Details"])
    st.table(revenue_modelling_df)












    st.divider()  # 최신 Streamlit
    # st.markdown("---")  # 혹은 이 방식도 가능

    st.header("Revenue Prediction Model Performance Evaluation")

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
    st.header("Customer Churn + Revenue Forecasting (Supabase Integration)")

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
    engine = create_engine( "postgresql://postgres:Nwk5JYywxV3ATT8M@db.fjaxvaegmtbsyogavuzy.supabase.co:5432/postgres" )

    # ---------------------------
    # 4. Streamlit UI
    # ---------------------------
    uploaded_file = st.file_uploader("Customer CSV upload", type="csv")

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

        # 📊 클러스터별 평균으로 Risk/Value 자동 라벨링
        cluster_summary = df.groupby("Cluster")[["churn_prob", "MonthlyCharges"]].mean()
        risk_threshold = df["churn_prob"].mean()
        value_threshold = df["MonthlyCharges"].mean()

        def auto_label(cluster):
            row = cluster_summary.loc[cluster]
            risk = "High Risk" if row["churn_prob"] >= risk_threshold else "Low Risk"
            value = "High Value" if row["MonthlyCharges"] >= value_threshold else "Low Value"
            return f"{risk} & {value}"

        df["cluster_label"] = df["Cluster"].apply(auto_label)

        # (5) base_message 생성
        base_messages = {
            "High Risk & High Value": "Exclusive premium offers to retain our top customers",
            "High Risk & Low Value": "Special discount to prevent churn at minimal cost",
            "Low Risk & High Value": "VIP thank-you campaign for loyal high-value customers",
            "Low Risk & Low Value": "Customer feedback request to strengthen relationships",
        }
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

    st.header("Next Steps & Open Challenges")

    st.markdown("""
- **Data expansion**: Include call center logs, customer support chats, and usage data.  
- **Real-time integration**: Connect models directly with CRM for live churn alerts.  
- **A/B testing**: Validate the effectiveness of personalized retention campaigns.  
- **Churn model optimization**: Explore XGBoost/LightGBM, better calibration, and automated threshold tuning.  
- **Model robustness**: How well does the model generalize across new customer cohorts?  
- **Ethics & fairness**: Could targeting strategies unintentionally bias or exclude groups?  
    """)

    st.divider()

    st.header("Long-term Vision")
    st.markdown("""
- Build **automation pipelines** with high-performing models beyond churn/revenue.  
- Enable **1 person to deliver the productivity of 10** through intelligent automation.  
- Move toward a future where **data-driven decision-making** is seamlessly embedded in daily operations.  
    """)

    st.markdown(
        """
        <sub>Note: This project is a first step toward scaling intelligent automation across business functions.</sub>
        """,
        unsafe_allow_html=True
    )

















