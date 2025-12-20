# src/dashboard/app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from utils import check_api_status, predict_text, predict_csv

# 1. Cấu hình trang (Phải để đầu tiên)
st.set_page_config(
    page_title="ViHOS Admin Panel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Sidebar - Menu điều hướng
with st.sidebar:
    st.title("🛡️ ViHOS Control")
    st.markdown("---")

    # Kiểm tra trạng thái Server
    is_live, info = check_api_status()
    if is_live:
        st.success(f"🟢 API Online ({info.get('device', 'UNKNOWN')})")
    else:
        st.error("🔴 API Offline")
        st.warning("Hãy chạy: `python src/api/server.py`")

    st.markdown("---")
    menu = st.radio("Menu", ["Dashboard & Live Scan", "Batch File Scanner"])

    st.markdown("---")
    st.info("System Administrator Mode")

# 3. Giao diện chính
if menu == "Dashboard & Live Scan":
    st.header("📡 Live Monitoring Console")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Kiểm tra nhanh (Quick Test)")
        input_text = st.text_area("Nhập nội dung cần kiểm duyệt:", height=150, placeholder="Ví dụ: Mày ngu quá...")

        if st.button("Quét ngay (Scan)", type="primary"):
            if not input_text.strip():
                st.warning("Vui lòng nhập nội dung!")
            elif not is_live:
                st.error("Không thể kết nối đến Backend API!")
            else:
                with st.spinner("AI đang phân tích..."):
                    result = predict_text(input_text)

                # Hiển thị kết quả
                if result.get("label") == "TOXIC":
                    st.error(f"❌ PHÁT HIỆN ĐỘC HẠI (TOXIC)")
                else:
                    st.success(f"✅ NỘI DUNG SẠCH (CLEAN)")

                # Chi tiết JSON
                st.json(result)

    with col2:
        st.subheader("Hướng dẫn Sysadmin")
        st.markdown("""
        - **Toxic:** Bao gồm Hate Speech, Offensive, Chửi thề.
        - **Clean:** Nội dung an toàn.
        - **Confidence:** Độ tin cậy của Model AI.
        """)
        st.markdown("💡 *Mẹo: Nếu Confidence < 70%, cần người duyệt lại.*")

elif menu == "Batch File Scanner":
    st.header("📂 Batch Log Scanner")
    st.markdown("Upload file log chat (.csv, .xlsx) để quét hàng loạt.")

    uploaded_file = st.file_uploader("Chọn file dữ liệu", type=["csv", "xlsx"])

    if uploaded_file:
        # Đọc file
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.dataframe(df.head())

            # Chọn cột chứa text
            text_col = st.selectbox("Chọn cột chứa nội dung chat:", df.columns)

            if st.button("Bắt đầu Quét (Start Batch Job)"):
                if not is_live:
                    st.error("API Offline!")
                else:
                    with st.spinner(f"Đang xử lý {len(df)} dòng... vui lòng chờ."):
                        # Gọi hàm xử lý
                        result_df = predict_csv(df, text_col)

                    st.success("✅ Đã xử lý xong!")

                    # Thống kê & Biểu đồ
                    c1, c2 = st.columns(2)
                    with c1:
                        st.subheader("Kết quả chi tiết")
                        st.dataframe(result_df)

                    with c2:
                        st.subheader("Thống kê tỉ lệ")
                        counts = result_df['Label'].value_counts()
                        fig = px.pie(
                            names=counts.index,
                            values=counts.values,
                            title="Tỷ lệ Nội dung Độc hại",
                            color_discrete_map={"TOXIC": "red", "CLEAN": "green", "ERROR": "gray"}
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Nút tải về
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Tải báo cáo (.csv)",
                        csv,
                        "vihos_report.csv",
                        "text/csv",
                        key='download-csv'
                    )
        except Exception as e:
            st.error(f"Lỗi đọc file: {e}")