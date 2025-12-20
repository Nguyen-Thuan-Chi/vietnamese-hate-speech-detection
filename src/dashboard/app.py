# src/dashboard/app.py
import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path
import plotly.express as px

# ===== PATH SETUP =====
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
sys.path.append(str(ROOT_DIR))

# ===== PAGE CONFIG (PHẢI Ở ĐẦU) =====
st.set_page_config(
    page_title="ViHOS Admin Panel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== LOAD MODEL (HF SPACE – CPU ONLY) =====
@st.cache_resource
def load_model():
    try:
        from src.services.predictor import HateSpeechPredictor
        model_path = ROOT_DIR / "models" / "phobert_epoch_3.pth"
        return HateSpeechPredictor(str(model_path), device="cpu")
    except Exception as e:
        st.error(f"❌ Không load được model: {e}")
        st.stop()

predictor = load_model()

# ===== HELPER FUNCTIONS =====
def predict_text_local(text: str):
    return predictor.predict(text)

def predict_csv_local(df: pd.DataFrame, text_col: str):
    results = []
    for _, row in df.iterrows():
        try:
            res = predictor.predict(str(row[text_col]))
            results.append({
                **row,
                "Label": res["label"],
                "Confidence": res["confidence"]
            })
        except Exception:
            results.append({
                **row,
                "Label": "ERROR",
                "Confidence": "0%"
            })
    return pd.DataFrame(results)

# ===== SIDEBAR =====
with st.sidebar:
    st.title("🛡️ ViHOS Control")
    st.markdown("---")
    st.success("🟢 Model Online (CPU)")
    st.markdown("---")
    menu = st.radio("Menu", ["Dashboard & Live Scan", "Batch File Scanner"])
    st.markdown("---")
    st.info("Running on Hugging Face Spaces")

# ===== MAIN UI =====
if menu == "Dashboard & Live Scan":
    st.header("📡 Live Monitoring Console")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Kiểm tra nhanh (Quick Test)")
        input_text = st.text_area(
            "Nhập nội dung cần kiểm duyệt:",
            height=150,
            placeholder="Ví dụ: Mày ngu quá..."
        )

        if st.button("Quét ngay (Scan)", type="primary"):
            if not input_text.strip():
                st.warning("Vui lòng nhập nội dung!")
            else:
                with st.spinner("AI đang phân tích..."):
                    result = predict_text_local(input_text)

                if result.get("label") == "TOXIC":
                    st.error("❌ PHÁT HIỆN ĐỘC HẠI (TOXIC)")
                else:
                    st.success("✅ NỘI DUNG SẠCH (CLEAN)")

                st.json(result)

    with col2:
        st.subheader("Hướng dẫn Sysadmin")
        st.markdown("""
        - **TOXIC:** Hate Speech, Offensive, Chửi thề
        - **CLEAN:** Nội dung an toàn
        - **Confidence:** Độ tin cậy của model
        """)
        st.markdown("💡 *Confidence < 70% → nên duyệt thủ công*")

elif menu == "Batch File Scanner":
    st.header("📂 Batch Log Scanner")
    st.markdown("Upload file chat (.csv, .xlsx) để quét hàng loạt.")

    uploaded_file = st.file_uploader(
        "Chọn file dữ liệu",
        type=["csv", "xlsx"]
    )

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.dataframe(df.head())

            text_col = st.selectbox(
                "Chọn cột chứa nội dung chat:",
                df.columns
            )

            if st.button("Bắt đầu Quét (Start Batch Job)"):
                with st.spinner(f"Đang xử lý {len(df)} dòng..."):
                    result_df = predict_csv_local(df, text_col)

                st.success("✅ Đã xử lý xong!")

                c1, c2 = st.columns(2)

                with c1:
                    st.subheader("Kết quả chi tiết")
                    st.dataframe(result_df)

                with c2:
                    st.subheader("Thống kê tỉ lệ")
                    counts = result_df["Label"].value_counts()
                    fig = px.pie(
                        names=counts.index,
                        values=counts.values,
                        title="Tỷ lệ Nội dung Độc hại",
                        color_discrete_map={
                            "TOXIC": "red",
                            "CLEAN": "green",
                            "ERROR": "gray"
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)

                csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Tải báo cáo (.csv)",
                    csv,
                    "vihos_report.csv",
                    "text/csv"
                )

        except Exception as e:
            st.error(f"Lỗi xử lý file: {e}")
