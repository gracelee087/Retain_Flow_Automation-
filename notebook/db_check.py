import sqlite3
import pandas as pd

conn = sqlite3.connect("customers.db")

tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
print("📌 테이블 목록:\n", tables)

df = pd.read_sql("SELECT * FROM predictions LIMIT 5;", conn)
print("\n📌 predictions 테이블 샘플:\n", df.head())

conn.close()

# python notebook/db_check.py 터미널에 치면 db 목록 나옴. 
# 근데 어차피 customers.db가 streamlit으로 인해서 생기기 때문에 굳이 쓸 필요는 없음.
