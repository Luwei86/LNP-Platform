import streamlit as st
import pandas as pd
import numpy as np
import pickle
import traceback

from rdkit import Chem
from mordred import Calculator, descriptors

# =========================================================
# 页面配置
# =========================================================
st.set_page_config(
    page_title="LNP AI Platform",
    page_icon="🧬",
    layout="wide"
)

# =========================================================
# 初始化 Mordred
# =========================================================
@st.cache_resource
def load_calculator():

    calc = Calculator(
        descriptors,
        ignore_3D=True
    )

    return calc

calc = load_calculator()

# =========================================================
# 加载模型
# =========================================================
@st.cache_resource
def load_model():

    with open("xgboost_justify_model.pkl", "rb") as f:

        model, label_encoder = pickle.load(f)

    return model, label_encoder

model, label_encoder = load_model()

# =========================================================
# 加载特征列表
# =========================================================
@st.cache_resource
def load_features():

    with open("model_features.pkl", "rb") as f:

        feature_names = pickle.load(f)

    return feature_names

feature_names = load_features()

# =========================================================
# SMILES → 描述符
# =========================================================
def smiles_to_descriptors(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:

        raise ValueError("Invalid SMILES")

    # 计算描述符
    df = calc.pandas([mol])

    # 转数值
    df = df.apply(
        pd.to_numeric,
        errors='coerce'
    )

    # 替换 inf
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # 删除全空列
    df = df.dropna(
        axis=1,
        how='all'
    )

    # 缺失值填充
    df = df.fillna(
        df.mean()
    )

    return df

# =========================================================
# 顶部导航栏
# =========================================================
col1, col2, col3, col4, col5 = st.columns([1, 0.5, 6, 0.5, 1])

with col1:

    st.image(
        "logo.svg",
        width=120
    )

with col3:

    st.markdown(
        """
        <div style="
            text-align:center;
            max-width:900px;
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

    st.image(
        "logo2.png",
        width=150
    )

st.markdown(
    "<hr style='margin-top:5px; margin-bottom:10px;'>",
    unsafe_allow_html=True
)

# =========================================================
# 页面布局
# =========================================================
left, right = st.columns([1, 2])

# =========================================================
# 左侧控制面板
# =========================================================
with left:

    st.markdown("## ⚙️ Control Panel")

    mode = st.radio(
        "选择模式",
        [
            "单分子预测",
            "批量预测"
        ]
    )

    smiles_input = None
    uploaded_file = None

    if mode == "单分子预测":

        smiles_input = st.text_input(
            "SMILES",
            "CCOCC"
        )

    else:

        uploaded_file = st.file_uploader(
            "上传Excel/CSV（第一列Name，第二列SMILES）",
            type=["xlsx", "csv"]
        )

    predict_btn = st.button(
        "🚀 开始预测",
        use_container_width=True
    )

# =========================================================
# 右侧结果面板
# =========================================================
with right:

    st.markdown("## 📊 Result Panel")

    # =====================================================
    # 单分子预测
    # =====================================================
    if mode == "单分子预测" and predict_btn:

        try:

            # 生成描述符
            df_desc = smiles_to_descriptors(
                smiles_input
            )

            # 对齐模型特征
            X = df_desc.reindex(
                columns=feature_names,
                fill_value=0
            )

            X = X.astype(float)

            # 预测
            pred = model.predict(X)

            label = label_encoder.inverse_transform(pred)[0]

            # 概率
            prob = model.predict_proba(X).max()

            # 显示结果
            st.success(
                f"🎯 Prediction: {label}"
            )

            st.info(
                f"📊 Confidence: {prob:.3f}"
            )

            # 显示描述符
            with st.expander("Molecular Descriptors"):

                st.dataframe(
                    df_desc,
                    use_container_width=True
                )

        except Exception:

            st.error("❌ Prediction Error")

            st.code(
                traceback.format_exc()
            )

    # =====================================================
    # 批量预测
    # =====================================================
    if (
        mode == "批量预测"
        and predict_btn
        and uploaded_file is not None
    ):

        try:

            # 读取文件
            if uploaded_file.name.endswith(".csv"):

                df_input = pd.read_csv(
                    uploaded_file
                )

            else:

                df_input = pd.read_excel(
                    uploaded_file
                )

            names = df_input.iloc[:, 0]
            smiles_list = df_input.iloc[:, 1]

            results = []

            progress = st.progress(0)

            # 批量处理
            for i, smi in enumerate(smiles_list):

                try:

                    # 描述符
                    df_desc = smiles_to_descriptors(
                        str(smi)
                    )

                    # 对齐特征
                    X = df_desc.reindex(
                        columns=feature_names,
                        fill_value=0
                    )

                    X = X.astype(float)

                    # 预测
                    pred = model.predict(X)

                    label = label_encoder.inverse_transform(
                        pred
                    )[0]

                    # 概率
                    prob = model.predict_proba(X).max()

                    results.append([
                        names[i],
                        smi,
                        label,
                        round(float(prob), 4)
                    ])

                except Exception as e:

                    results.append([
                        names[i],
                        smi,
                        "ERROR",
                        str(e)
                    ])

                # 更新进度条
                progress.progress(
                    (i + 1) / len(smiles_list)
                )

            # 结果表
            df_result = pd.DataFrame(
                results,
                columns=[
                    "Name",
                    "SMILES",
                    "Prediction",
                    "Confidence/Error"
                ]
            )

            st.success(
                "🎯 Batch Prediction Completed"
            )

            st.dataframe(
                df_result,
                use_container_width=True
            )

            # 下载按钮
            csv = df_result.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Result CSV",
                csv,
                "prediction_result.csv",
                "text/csv",
                use_container_width=True
            )

        except Exception:

            st.error("❌ Batch Prediction Error")

            st.code(
                traceback.format_exc()
            )
