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
st.set_page_config(page_title="Jリーグで勝点を取る", layout="wide")
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

# ---------------- サイドバー：ファイル読み込み ----------------
st.sidebar.header("ファイル読み込み")
uploaded = st.sidebar.file_uploader(
    "Excel（.xlsx）または CSV（.csv）を選んでください", type=["xlsx", "csv"]
)

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

    # 数値列の抽出
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

                    # ◆ 散布図サイズを 2/3 に縮小（元 5.2×3.6 → 約 3.47×2.40 inch）
                    fig, ax = plt.subplots(figsize=(3.47, 2.40))
                    ax.scatter(x, y, s=22, alpha=0.85)
                    xx = np.linspace(np.min(x), np.max(x), 200)
                    yy = slope * xx + intercept
                    ax.plot(xx, yy)
                    ax.set_xlabel(f"{feature_col}（説明変数：横軸）")
                    ax.set_ylabel(f"{target_col}（目的変数：縦軸）")
                    ax.set_title("散布図と回帰直線")
                    st.pyplot(fig, use_container_width=False)

                    # --- 散布図の下に、回帰情報を表示 ---
                    st.markdown("#### 回帰の結果（散布図の下に表示）")
                    st.write(f"- 回帰式：  **y = {slope:.4g} × x + {intercept:.4g}**")
                    st.write(f"- 相関係数 r： **{r_value:.4f}**")
                    st.write(f"- 決定係数 R²： **{r2:.4f}**")

                    # --- 予測用の x 入力 → y を計算して表示 ---
                    st.markdown("#### 任意の x を入れて、予測 y を計算")
                    xmin, xmax = float(np.min(x)), float(np.max(x))
                    x_default = float(np.mean(x))
                    col_l, col_r = st.columns([1, 1])
                    with col_l:
                        st.caption(f"参考：データの x 範囲 ≈ [{xmin:.4g}, {xmax:.4g}]")
                    with col_r:
                        st.caption(f"平均 x ≈ {x_default:.4g}")

                    step = (xmax - xmin) / 100 if xmax > xmin else 1.0
                    x_input = st.number_input(
                        "説明変数 x の値を入力", value=x_default, step=step, format="%.6f"
                    )
                    y_pred = slope * x_input + intercept
                    st.success(f"予測された  **{target_col} (y)**  の値： **{y_pred:.6g}**")

        # ---------------- 散布図行列（各セルに回帰直線・回帰式・rを表示） ----------------
        with tab2:
            st.markdown("**散布図行列に含める変数に☑してください**（数値列のみ表示）。")
            default_cols = numeric_cols[:min(4, len(numeric_cols))]
            selected_cols = st.multiselect(
                "散布図行列に入れる列を選択（2〜6列程度を推奨）",
                options=numeric_cols, default=default_cols
            )

            if len(selected_cols) >= 2:
                plot_df = df[selected_cols].dropna().astype(float)
                if len(plot_df) < 2:
                    st.warning("有効なデータ点が少なすぎます。別の列を選んでください。")
                else:
                    n = len(selected_cols)
                    # 列数に応じてセルサイズを自動調整（見やすさ優先）
                    if n <= 3:
                        cell = 2.2
                    elif n <= 4:
                        cell = 2.0
                    elif n <= 5:
                        cell = 1.8
                    else:
                        cell = 1.6
                    fig_w, fig_h = cell * n, cell * n

                    # 先に軸を作成してから scatter_matrix を当てる
                    fig, axs = plt.subplots(n, n, figsize=(fig_w, fig_h))
                    axes = scatter_matrix(
                        plot_df, ax=axs, diagonal='hist', alpha=0.7, s=12
                    )

                    # 各オフ対角セルに回帰直線・式・r を描画
                    for i, y_name in enumerate(selected_cols):      # 行（縦）= y
                        for j, x_name in enumerate(selected_cols):  # 列（横）= x
                            if i == j:
                                # 対角はヒストグラムのまま
                                continue
                            ax = axes[i, j]
                            x = plot_df[x_name].values
                            y = plot_df[y_name].values

                            # 定数列などで回帰ができない場合はスキップ
                            if np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
                                continue

                            reg = linregress(x, y)
                            slope, intercept, r = reg.slope, reg.intercept, reg.rvalue

                            # 回帰直線
                            x_min, x_max = np.min(x), np.max(x)
                            xx = np.linspace(x_min, x_max, 50)
                            yy = slope * xx + intercept
                            ax.plot(xx, yy, linewidth=1)

                            # 簡潔な注記（左上に小さく）
                            eq = f"y={slope:.2g}x+{intercept:.2g}\nr={r:.2f}"
                            ax.text(
                                0.03, 0.97, eq,
                                transform=ax.transAxes, ha="left", va="top",
                                fontsize=7, bbox=dict(boxstyle='round', alpha=0.08)
                            )

                    # 端のラベルだけ残して可読性を確保
                    for i, row_name in enumerate(selected_cols):
                        for j, col_name in enumerate(selected_cols):
                            ax = axes[i, j]
                            if i == n - 1:
                                ax.set_xlabel(col_name, fontsize=9)
                            else:
                                ax.set_xlabel("")
                                ax.set_xticklabels([])
                            if j == 0:
                                ax.set_ylabel(row_name, fontsize=9)
                            else:
                                ax.set_ylabel("")
                                ax.set_yticklabels([])

                    plt.tight_layout(pad=0.6)
                    st.pyplot(fig, use_container_width=False)
            else:
                st.info("2つ以上の列を選ぶと散布図行列を表示します。")
else:
    st.info("左のサイドバーからデータファイルを読み込んでください（.xlsx / .csv）。")
