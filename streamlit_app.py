import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

# -------------- 基本設定 --------------
st.set_page_config(page_title="相関と回帰の学習ツール", layout="wide")

st.title("相関と回帰の学習ツール")
st.caption("高校生向け：ファイルを読み込み、散布図と回帰、相関行列を確認します。")

with st.expander("このツールでできること（最初に読んでください）", expanded=True):
    st.markdown(
        """
- **目的変数**：予測したい量・結果のことです（例：売上）。
- **説明変数**：目的変数に影響しそうな要因のことです（例：広告費）。
- **手順**
    1. 画面左の *ファイル読み込み* から、`.xlsx`（Excel）または`.csv`をアップロードします。
    2. 「散布図と回帰」タブで、**目的変数**（縦軸）と **説明変数**（横軸）を選びます。
       - 散布図の上に **回帰式**, **相関係数 r**, **決定係数 R²** を表示します。
    3. 「相関行列」タブで、比較したい変数に☑を入れて、**相関行列**を作ります。
        """
    )

# -------------- サイドバー：ファイル読み込み --------------
st.sidebar.header("ファイル読み込み")

uploaded = st.sidebar.file_uploader("Excel（.xlsx）または CSV（.csv）を選んでください", type=["xlsx", "csv"])

df = None
if uploaded is not None:
    file_name = uploaded.name.lower()
    try:
        if file_name.endswith(".xlsx"):
            # シート選択に対応
            xls = pd.ExcelFile(uploaded)
            sheet = st.sidebar.selectbox("シートを選んでください", options=xls.sheet_names, index=0)
            df = pd.read_excel(xls, sheet_name=sheet)
        else:
            # CSV の読み込み
            df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"ファイルの読み込みでエラーが発生しました: {e}")

# -------------- データ表示 --------------
if df is not None:
    st.subheader("読み込んだデータ（スクロールで全体を確認できます）")
    st.dataframe(df, use_container_width=True, height=420)

    # 数値列のみを候補にする
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) == 0:
        st.warning("数値列が見つかりませんでした。数値データを含むファイルを読み込んでください。")
    else:
        tab1, tab2 = st.tabs(["散布図と回帰", "相関行列"])

        # ---------------- 散布図と回帰 ----------------
        with tab1:
            st.markdown("**目的変数（縦軸）**：予測したい結果を選んでください。")
            target_col = st.selectbox("目的変数を選択", options=numeric_cols, index=0, key="target")
            st.markdown("**説明変数（横軸）**：目的に影響しそうな要因を1つ選んでください。")
            feature_col = st.selectbox("説明変数を選択", options=[c for c in numeric_cols if c != target_col], index=0, key="feature")

            if target_col and feature_col:
                # 欠損値の除去
                data = df[[feature_col, target_col]].dropna()
                x = data[feature_col].astype(float).values
                y = data[target_col].astype(float).values

                if len(x) < 2:
                    st.warning("有効なデータ点が少なすぎます。別の列を選んでください。")
                else:
                    reg = linregress(x, y)
                    slope = reg.slope
                    intercept = reg.intercept
                    r_value = reg.rvalue
                    r2 = r_value ** 2

                    # 散布図と回帰直線の描画
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.scatter(x, y, alpha=0.8)
                    xx = np.linspace(np.min(x), np.max(x), 200)
                    yy = slope * xx + intercept
                    ax.plot(xx, yy)
                    ax.set_xlabel(f"{feature_col}（説明変数：横軸）")
                    ax.set_ylabel(f"{target_col}（目的変数：縦軸）")
                    ax.set_title("散布図と回帰直線")

                    # グラフ内に数式等を注記
                    eq_text = f"回帰式: y = {slope:.4g} x + {intercept:.4g}\n相関係数 r = {r_value:.4f}\n決定係数 R² = {r2:.4f}"
                    ax.text(0.02, 0.98, eq_text, transform=ax.transAxes,
                            ha="left", va="top", bbox=dict(boxstyle="round", alpha=0.1))

                    st.pyplot(fig, use_container_width=True)

        # ---------------- 相関行列 ----------------
        with tab2:
            st.markdown("**比較したい変数に☑してください**（数値列のみ表示します）。")
            default_cols = numeric_cols[:min(5, len(numeric_cols))]
            selected_cols = st.multiselect("相関を見たい列を選択", options=numeric_cols, default=default_cols)

            if len(selected_cols) >= 2:
                corr = df[selected_cols].corr(method="pearson")
                st.subheader("相関行列（Pearson）")
                st.dataframe(corr, use_container_width=True, height=420)
            else:
                st.info("2つ以上の列を選ぶと相関行列を表示します。")
else:
    st.info("左のサイドバーからデータファイルを読み込んでください（.xlsx / .csv）。")
