# src/dashboard/app.py
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import plotly.express as px
from typing import Dict, List, Union

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

# ===== LOAD SERVICES =====
from src.services.highlighter import ToxicHighlighter, KeywordHighlighter
from src.services.feedback import FeedbackManager
from src.services.explainer import LimeTextExplainerService, create_predict_proba_wrapper

highlighter = ToxicHighlighter()
keyword_highlighter = KeywordHighlighter()
feedback_manager = FeedbackManager(ROOT_DIR / "data" / "feedback.csv")

# ===== LIME EXPLAINER (XAI) =====
@st.cache_resource
def load_explainer():
    return LimeTextExplainerService()

lime_explainer = load_explainer()

# Minimum word count threshold for LIME explanation
MIN_WORDS_FOR_LIME = 5

# ===== HELPER FUNCTIONS =====
def predict_text_local(text: str) -> Dict[str, Union[str, List[Dict[str, Union[int, str]]]]]:
    """
    Predict hate speech classification.

    Note: Current model is sentence-level only. The 'spans' field will always
    be empty. UI uses keyword-based fallback for highlighting toxic content.

    Returns dict with:
        - label: "TOXIC" or "CLEAN"
        - confidence: confidence percentage string
        - spans: Always empty (sentence-level model)
    """
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
                    # Store result in session state for feedback
                    st.session_state["last_prediction"] = {
                        "text": input_text,
                        "result": result
                    }

                if result.get("label") == "TOXIC":
                    st.error("❌ PHÁT HIỆN ĐỘC HẠI (TOXIC)")
                else:
                    st.success("✅ NỘI DUNG SẠCH (CLEAN)")

                # Display highlighted text with toxic spans
                spans = result.get("spans", [])
                label = result.get("label", "CLEAN")

                if label == "TOXIC":
                    # Keyword-based highlighting (sentence-level model has no span data)
                    st.markdown("**Từ khóa đáng ngờ được đánh dấu:**")
                    highlighted_html = keyword_highlighter.highlight(input_text)
                    st.markdown(highlighted_html, unsafe_allow_html=True)
                    st.caption(
                        "⚠️ *Lưu ý: Model phân loại ở cấp độ câu, không xác định chính xác vị trí độc hại. "
                        "Các từ được đánh dấu dựa trên danh sách từ khóa tham khảo.*"
                    )

                    # ===== LIME EXPLANATION (XAI) =====
                    st.markdown("---")
                    st.markdown("**🔍 Phân tích XAI (LIME):**")

                    # Get preprocessed text for word count check
                    clean_text = result.get("text_clean", input_text)
                    word_count = len(clean_text.split())

                    if word_count >= MIN_WORDS_FOR_LIME:
                        with st.spinner("Đang phân tích nguyên nhân dự đoán..."):
                            try:
                                predict_proba_fn = create_predict_proba_wrapper(predictor)
                                word_weights = lime_explainer.explain(
                                    text=clean_text,
                                    predict_proba_fn=predict_proba_fn,
                                    num_features=10,
                                    num_samples=500,
                                    label_index=1  # TOXIC
                                )
                            except Exception as e:
                                word_weights = []
                                st.warning(f"Lỗi khi phân tích LIME: {e}")

                        if word_weights:
                            # Render bar chart for word contributions
                            df_weights = pd.DataFrame(word_weights)
                            df_weights["color"] = df_weights["weight"].apply(
                                lambda w: "Góp phần TOXIC" if w > 0 else "Góp phần CLEAN"
                            )
                            fig = px.bar(
                                df_weights,
                                x="weight",
                                y="word",
                                orientation="h",
                                color="color",
                                color_discrete_map={
                                    "Góp phần TOXIC": "#ff4b4b",
                                    "Góp phần CLEAN": "#21c354"
                                },
                                title="Mức độ ảnh hưởng của từng từ đến dự đoán TOXIC"
                            )
                            fig.update_layout(
                                yaxis=dict(autorange="reversed"),
                                showlegend=True,
                                height=350
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption(
                                "📊 *Thanh đỏ: từ làm tăng khả năng TOXIC. "
                                "Thanh xanh: từ làm giảm khả năng TOXIC.*"
                            )
                        else:
                            st.info(
                                "ℹ️ Câu quá ngắn hoặc quá độc hại để phân tích XAI chi tiết. "
                                "LIME không thể tạo giải thích ổn định cho văn bản này."
                            )
                    else:
                        st.info(
                            f"ℹ️ Câu quá ngắn ({word_count} từ < {MIN_WORDS_FOR_LIME} từ) "
                            "để phân tích XAI chi tiết. Cần ít nhất 5 từ để LIME hoạt động ổn định."
                        )

                st.json(result)

        # ===== FEEDBACK UI =====
        if "last_prediction" in st.session_state:
            st.markdown("---")
            st.subheader("📝 Phản hồi (Feedback)")

            last_pred = st.session_state["last_prediction"]
            pred_text = last_pred["text"]
            pred_result = last_pred["result"]
            pred_spans = pred_result.get("spans", [])

            feedback_col1, feedback_col2 = st.columns(2)

            with feedback_col1:
                if st.button("✅ Dự đoán chính xác", key="feedback_correct"):
                    feedback_manager.save_feedback(
                        text=pred_text,
                        spans=pred_spans,
                        user_feedback="correct"
                    )
                    st.success("Cảm ơn phản hồi của bạn!")
                    del st.session_state["last_prediction"]
                    st.rerun()

            with feedback_col2:
                if st.button("❌ Dự đoán không chính xác", key="feedback_incorrect"):
                    st.session_state["show_correction_input"] = True

            # Show correction input if user marked prediction as incorrect
            if st.session_state.get("show_correction_input", False):
                correction_text = st.text_area(
                    "Vui lòng mô tả lỗi hoặc nhập đoạn văn bản độc hại đúng:",
                    placeholder="Ví dụ: Đoạn 'xyz' là độc hại nhưng không được phát hiện...",
                    key="correction_input"
                )

                if st.button("Gửi phản hồi", key="submit_correction"):
                    if correction_text.strip():
                        feedback_manager.save_feedback(
                            text=pred_text,
                            spans=pred_spans,
                            user_feedback=f"incorrect: {correction_text}"
                        )
                        st.success("Cảm ơn phản hồi chi tiết của bạn!")
                        st.session_state["show_correction_input"] = False
                        del st.session_state["last_prediction"]
                        st.rerun()
                    else:
                        st.warning("Vui lòng nhập mô tả lỗi!")

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
