import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from matplotlib import font_manager as fm, rcParams
from pathlib import Path

# ---------------- フォント設定（GitHubの ./fonts/SourceHanCodeJP-Regular.otf を使用） ----------------
def setup_japanese_font():
    font_file = Path(__file__).parent / "fonts" / "SourceHanCodeJP-Regular.otf"
    try:
        if font_file.exists():
            fm.fontManager.addfont(str(font_file))
            rcParams["font.family"] = "Source Han Code JP"  # 登録名
        rcParams["axes.unicode_minus"] = False  # マイナス記号の文字化け防止
    except Exception as e:
        st.warning(f"日本語フォントの設定に失敗しました: {e}")

setup_japanese_font()

# ---------------- 基本設定 ----------------
st.set_page_config(page_title="相関と回帰の学習ツール", layout="wide")
st.title("相関と回帰の学習ツール")
st.caption("高校生向け：ファイルを読み込み、散布図と回帰、散布図行列を確認します。")

# ---------------- タブの色（選択中のタブを色付け） ----------------
tab_css = """
<style>
.stTabs [data-baseweb="tab"] {
    background-color: #f7f7f9;
    padding: 6px 14px;
    border-radius: 8px 8px 0 0;
    margin-right: 6px;
    font-weight: 600;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background-color: #e8f0fe;
    border-bottom: 2px solid #1f77b4;
    color: black;
}
</style>
"""
st.markdown(tab_css, unsafe_allow_html=True)

with st.expander("このツールでできること（最初に読んでください）", expanded=True):
    st.markdown(
        """
- **目的変数**：予測したい量・結果のことです（例：売上）。
- **説明変数**：目的変数に影響しそうな要因のことです（例：広告費）。
- **手順**
    1. 画面左の *ファイル読み込み* から、`.xlsx`（Excel）または`.csv`をアップロードします。
    2. 「散布図と回帰」タブで、**目的変数**（縦軸）と **説明変数**（横軸）を選びます。
       - グラフ内に **回帰式**, **相関係数 r**, **決定係数 R²** を表示します。
    3. 「散布図行列」タブで、比較したい変数に☑を入れて、**散布図行列**を作ります（相関の強さ・形を俯瞰）。
        """
    )

# ---------------- サイドバー：ファイル読み込み ----------------
st.sidebar.header("ファイル読み込み")
uploaded = st.sidebar.file_uploader(
    "Excel（.xlsx）または CSV（.csv）を選んでください", type=["xlsx", "csv"]
)

# グラフサイズと点サイズの設定（ユーザー調整可能）
st.sidebar.header("グラフ表示設定")
fig_w = st.sidebar.slider("図の横幅（インチ換算）", min_value=3.0, max_value=10.0, value=5.0, step=0.5)
fig_h = st.sidebar.slider("図の縦幅（インチ換算）", min_value=2.0, max_value=8.0, value=3.5, step=0.5)
marker_size = st.sidebar.slider("点の大きさ", min_value=5, max_value=80, value=24, step=1)
alpha = st.sidebar.slider("点の透明度", min_value=0.1, max_value=1.0, value=0.85, step=0.05)

df = None
if uploaded is not None:
    file_name = uploaded.name.lower()
    try:
        if file_name.endswith(".xlsx"):
            xls = pd.ExcelFile(uploaded)
            sheet = st.sidebar.selectbox("シートを選んでください", options=xls.sheet_names, index=0)
            df = pd.read_excel(xls, sheet_name=sheet)
        else:
            df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"ファイルの読み込みでエラーが発生しました: {e}")

# ---------------- データ表示 ----------------
if df is not None:
    st.subheader("読み込んだデータ（スクロールで全体を確認できます）")
    st.dataframe(df, use_container_width=True, height=360)

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) == 0:
        st.warning("数値列が見つかりませんでした。数値データを含むファイルを読み込んでください。")
    else:
        tab1, tab2 = st.tabs(["散布図と回帰", "散布図行列"])

        # ---------------- 散布図と回帰 ----------------
        with tab1:
            st.markdown("**目的変数（縦軸）**：予測したい結果を選んでください。")
            target_col = st.selectbox("目的変数を選択", options=numeric_cols, index=0, key="target")

            st.markdown("**説明変数（横軸）**：目的に影響しそうな要因を1つ選んでください。")
            feature_candidates = [c for c in numeric_cols if c != target_col] or numeric_cols
            feature_col = st.selectbox("説明変数を選択", options=feature_candidates, index=0, key="feature")

            if target_col and feature_col:
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

                    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                    ax.scatter(x, y, s=marker_size, alpha=alpha)
                    xx = np.linspace(np.min(x), np.max(x), 200)
                    yy = slope * xx + intercept
                    ax.plot(xx, yy)
                    ax.set_xlabel(f"{feature_col}（説明変数：横軸）")
                    ax.set_ylabel(f"{target_col}（目的変数：縦軸）")
                    ax.set_title("散布図と回帰直線")

                    eq_text = f"回帰式: y = {slope:.4g} x + {intercept:.4g}\n相関係数 r = {r_value:.4f}\n決定係数 R² = {r2:.4f}"
                    ax.text(0.02, 0.98, eq_text, transform=ax.transAxes, ha="left", va="top",
                            bbox=dict(boxstyle='round', alpha=0.1))

                    st.pyplot(fig, use_container_width=True)

        # ---------------- 散布図行列（相関行列の代わり） ----------------
        with tab2:
            st.markdown("**散布図行列に含める変数に☑してください**（数値列のみ表示）。")
            default_cols = numeric_cols[:min(4, len(numeric_cols))]
            selected_cols = st.multiselect("散布図行列に入れる列を選択（2〜8列程度を推奨）",
                                           options=numeric_cols, default=default_cols)

            if len(selected_cols) >= 2:
                cell_size = st.slider("セル一辺（インチ換算）", 1.5, 3.5, 2.0, 0.1)
                n = len(selected_cols)
                fig_size = (cell_size * n, cell_size * n)

                plot_df = df[selected_cols].dropna().astype(float)
                if len(plot_df) < 2:
                    st.warning("有効なデータ点が少なすぎます。別の列を選んでください。")
                else:
                    fig = plt.figure(figsize=fig_size)
                    axs = scatter_matrix(plot_df, figsize=fig_size, diagonal='hist', alpha=0.7, s=marker_size/2.0)
                    for i, col in enumerate(selected_cols):
                        for j in range(len(selected_cols)):
                            ax = axs[i, j]
                            if i == len(selected_cols) - 1:
                                ax.set_xlabel(selected_cols[j])
                            if j == 0:
                                ax.set_ylabel(col)
                    st.pyplot(fig, use_container_width=True)
            else:
                st.info("2つ以上の列を選ぶと散布図行列を表示します。")
else:
    st.info("左のサイドバーからデータファイルを読み込んでください（.xlsx / .csv）。")
