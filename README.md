# RetainFlow Automation: Customer Churn Prediction

## 프로젝트 개요
이 프로젝트는 고객 이탈 예측과 매출 예측을 통해 고객 유지 전략을 자동화하는 시스템입니다. Streamlit을 사용한 웹 애플리케이션과 Supabase 데이터베이스 연동을 통해 실시간 예측 및 데이터 저장이 가능합니다.

## 주요 기능
- **고객 이탈 예측**: 머신러닝 모델을 통한 고객 이탈 확률 예측
- **매출 예측**: 고객의 예상 수익 예측
- **고객 세그먼테이션**: 위험도와 가치에 따른 고객 분류
- **Supabase 연동**: 실시간 데이터 저장 및 관리
- **Streamlit Cloud 배포**: 웹 기반 인터페이스 제공

## 기술 스택
- **Frontend**: Streamlit
- **Backend**: Python, SQLAlchemy
- **Database**: Supabase (PostgreSQL)
- **ML**: scikit-learn, imbalanced-learn
- **Visualization**: matplotlib, seaborn

## Streamlit Cloud 배포 가이드

### 1. GitHub 저장소 준비
1. 이 프로젝트를 GitHub 저장소에 업로드
2. 모든 파일이 올바른 위치에 있는지 확인

### 2. Streamlit Cloud 설정
1. [Streamlit Cloud](https://share.streamlit.io/)에 접속
2. "New app" 클릭
3. GitHub 저장소 연결
4. 메인 파일 경로: `Streamlit_app.py`

### 3. 환경변수 설정
Streamlit Cloud의 Secrets Management에서 다음 환경변수를 설정:

```toml
[DATABASE_URL]
DATABASE_URL = "postgresql+psycopg2://postgres:YOUR_PASSWORD@db.fjaxvaegmtbsyogavuzy.supabase.co:5432/postgres?sslmode=require"
```

### 4. 필요한 파일 구조
```
├── Streamlit_app.py
├── requirements.txt
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml
├── notebook/
│   ├── pipeline_customer_churn_model.pkl
│   ├── pipeline_customer_revenue_model.pkl
│   ├── notebook/
│   │   ├── eda_insight/
│   │   └── modeling_insight/
│   └── revenue_insight/
└── pic.png
```

## 로컬 실행
```bash
pip install -r requirements.txt
streamlit run Streamlit_app.py
```

## 주의사항
- Supabase 데이터베이스 연결 정보는 환경변수로 관리
- 모델 파일과 이미지 파일은 상대 경로로 참조
- Streamlit Cloud에서는 절대 경로 사용 불가