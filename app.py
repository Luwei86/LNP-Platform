import streamlit as st
import pandas as pd
import pickle
import traceback
from padelpy import from_smiles

# =========================
# 页面配置
# =========================
st.set_page_config(
    page_title="LNP AI Platform",
    page_icon="🧬",
    layout="wide"
)

# =========================
# 顶部导航栏（紧凑版）
# =========================
col1, col2, col3, col4, col5 = st.columns([1, 0.5, 6, 0.5, 1])

with col1:
    st.image("logo.svg", width=120)

with col3:
    st.markdown(
        """
        <div style="
            text-align:center;
            max-width:800px;
            margin:auto;
        ">
            <h2 style="
                margin:0;
                padding:0;
                line-height:1.1;
            ">
            🧬 LNP / Lipid Molecular Prediction Platform
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )

with col5:
    st.image("logo2.png", width=150)

st.markdown("<hr style='margin-top:5px; margin-bottom:10px;'>", unsafe_allow_html=True)

# =========================
# 加载模型
# =========================
with open("xgboost_justify_model.pkl", "rb") as f:
    model, label_encoder = pickle.load(f)

# =========================
# 布局：左侧控制 + 右侧结果
# =========================
left, right = st.columns([1, 2])

# =========================================================
# 🧪 左侧控制面板
# =========================================================
with left:

    st.markdown("## ⚙️ Control Panel")

    mode = st.radio(
        "选择模式",
        ["单分子预测", "批量预测"]
    )

    smiles_input = None
    uploaded_file = None

    if mode == "单分子预测":
        smiles_input = st.text_input("SMILES", "CCOCC")

    else:
        uploaded_file = st.file_uploader(
            "上传Excel（第一列Name + 第二列SMILES）",
            type=["xlsx", "csv"]
        )

    predict_btn = st.button("🚀 开始预测")

# =========================================================
# 🧪 右侧结果面板
# =========================================================
with right:

    st.markdown("## 📊 Result Panel")

    # =========================
    # 单分子预测
    # =========================
    if mode == "单分子预测" and predict_btn:

        try:
            descriptors = from_smiles(smiles_input)
            df = pd.DataFrame([descriptors])

            df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

            features = model.get_booster().feature_names
            X = df.reindex(columns=features, fill_value=0).astype(float)

            pred = model.predict(X)
            label = label_encoder.inverse_transform(pred)[0]

            st.success(f"🎯 Prediction: {label}")

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X).max()
                st.info(f"📊 Confidence: {prob:.3f}")

            with st.expander("Descriptors"):
                st.dataframe(df)

        except Exception:
            st.error("❌ Error")
            st.code(traceback.format_exc())

    # =========================
    # 批量预测
    # =========================
    if mode == "批量预测" and predict_btn and uploaded_file:

        try:
            if uploaded_file.name.endswith(".csv"):
                df_input = pd.read_csv(uploaded_file)
            else:
                df_input = pd.read_excel(uploaded_file)

            names = df_input.iloc[:, 0]
            smiles_list = df_input.iloc[:, 1]

            results = []

            progress = st.progress(0)

            for i, smi in enumerate(smiles_list):

                try:
                    descriptors = from_smiles(str(smi))
                    df_desc = pd.DataFrame([descriptors])

                    df_desc = df_desc.apply(pd.to_numeric, errors="coerce").fillna(0)

                    features = model.get_booster().feature_names
                    X = df_desc.reindex(columns=features, fill_value=0).astype(float)

                    pred = model.predict(X)
                    label = label_encoder.inverse_transform(pred)[0]

                    if hasattr(model, "predict_proba"):
                        prob = model.predict_proba(X).max()
                    else:
                        prob = None

                    results.append([names[i], smi, label, prob])

                except Exception as e:
                    results.append([names[i], smi, "ERROR", str(e)])

                progress.progress((i + 1) / len(smiles_list))

            df_result = pd.DataFrame(
                results,
                columns=["Name", "SMILES", "Prediction", "Confidence/Error"]
            )

            st.success("🎯 Batch Prediction Completed")

            st.dataframe(df_result, use_container_width=True)

            csv = df_result.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇️ Download Result CSV",
                csv,
                "result.csv",
                "text/csv"
            )

        except Exception:
            st.error("❌ Batch Error")
            st.code(traceback.format_exc())
