import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet

# -------------------------
# アプリの基本設定
# -------------------------
st.set_page_config(page_title="高度テクニカル＆AI予測ダッシュボード", layout="wide")
st.title("📈 高度テクニカル＆AI予測ダッシュボード")

# -------------------------
# サイドバー（入力UI）
# -------------------------
st.sidebar.header("基本設定")

# 【変更部分】手入力からプルダウンメニューへ変更
ticker_options = {
    "金 (Gold)": "GC=F",
    "ドル円 (USD/JPY)": "JPY=X",
    "ポンド円 (GBP/JPY)": "GBPJPY=X",
    "ユーロ円 (EUR/JPY)": "EURJPY=X",
    "日経平均株価": "^N225",
    "S&P500 (米国株)": "^GSPC",
    "Apple (米国株)": "AAPL"
}
selected_ticker_name = st.sidebar.selectbox("銘柄を選択", list(ticker_options.keys()))
ticker_symbol = ticker_options[selected_ticker_name]

# 時間足の選択
interval_options = {
    "1日足": "1d",
    "1時間足": "1h",
    "15分足": "15m",
    "5分足": "5m"
}
selected_interval_label = st.sidebar.selectbox("時間足", list(interval_options.keys()))
interval = interval_options[selected_interval_label]

# Yahoo Financeの制限に対する注意書き
if interval in ["1h"]:
    st.sidebar.warning("1時間足のデータは過去730日分まで取得可能です。")
elif interval in ["15m", "5m"]:
    st.sidebar.warning(f"{selected_interval_label}のデータは過去60日分まで取得可能です。")

# 期間の設定
today = datetime.date.today()
if interval == "1d":
    default_start = today - pd.DateOffset(years=1)
elif interval == "1h":
    default_start = today - pd.DateOffset(months=6)
else: # 15m, 5m
    default_start = today - pd.DateOffset(days=30)

start_date = st.sidebar.date_input("開始日", default_start)
end_date = st.sidebar.date_input("終了日", today)

st.sidebar.markdown("---")
st.sidebar.subheader("テクニカル指標")
sma_short_window = st.sidebar.slider("短期SMA", 5, 50, 20)
sma_long_window = st.sidebar.slider("長期SMA", 20, 200, 50)
rsi_window = st.sidebar.slider("RSI期間", 7, 30, 14)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI予測 (Prophet)")
show_forecast = st.sidebar.checkbox("将来予測を表示する", value=False)
forecast_periods = st.sidebar.slider(f"予測する期間（{selected_interval_label}の数）", min_value=10, max_value=200, value=30, disabled=not show_forecast)

# Prophet用のfreq（頻度）マッピング
prophet_freq_mapping = {
    "1d": "D",
    "1h": "H",
    "15m": "15min",
    "5m": "5min"
}

# -------------------------
# メイン処理とデータ取得
# -------------------------
if ticker_symbol:
    with st.spinner(f'{selected_ticker_name} ({selected_interval_label}) のデータを取得中...'):
        try:
            # データの取得 (intervalを指定)
            df = yf.download(ticker_symbol, start=start_date, end=end_date, interval=interval)
            
            # yfinanceの新しいデータ構造を平坦化する処理
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            if df.empty:
                st.error("データが取得できませんでした。時間足の制限（短時間足は過去数ヶ月のみ）を超えている可能性があります。開始日を最近に設定してください。")
            else:
                # --- テクニカル指標の計算 ---
                df[f'SMA_{sma_short_window}'] = df['Close'].rolling(window=sma_short_window).mean()
                df[f'SMA_{sma_long_window}'] = df['Close'].rolling(window=sma_long_window).mean()

                delta = df['Close'].diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.ewm(alpha=1/rsi_window, adjust=False).mean()
                avg_loss = loss.ewm(alpha=1/rsi_window, adjust=False).mean()
                df['RSI'] = 100 - (100 / (1 + (avg_gain / avg_loss)))

                # --- AI予測 (Prophet) の計算 ---
                forecast = None
                if show_forecast:
                    with st.spinner('AIが将来の推移を計算中...'):
                        df_prophet = df[['Close']].reset_index()
                        
                        # カラム名の動的処理（日足は'Date'、日中足は'Datetime'になるため）
                        date_col = df_prophet.columns[0]
                        df_prophet = df_prophet.rename(columns={date_col: 'ds', 'Close': 'y'})
                        
                        # タイムゾーン情報の削除
                        if df_prophet['ds'].dt.tz is not None:
                            df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)

                        # モデルの学習パラメータ設定 (日足以外なら日次季節性を考慮)
                        daily_seasonality = True if interval != "1d" else False
                        yearly_seasonality = True if interval == "1d" else False
                        
                        m = Prophet(daily_seasonality=daily_seasonality, yearly_seasonality=yearly_seasonality)
                        m.fit(df_prophet)

                        # 未来の期間を作成
                        future = m.make_future_dataframe(periods=forecast_periods, freq=prophet_freq_mapping[interval])
                        
                        # 土日（株式・為替市場が閉まっている日）を予測から除外
                        future = future[future['ds'].dt.dayofweek < 5]
                        forecast = m.predict(future)

                # --- Plotlyでのグラフ描画 ---
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])

                # 【1段目】ローソク足チャートとSMA
                # 日中足の場合、インデックスがDatetime型になるための調整
                x_data = df.index.tz_localize(None) if df.index.tz is not None else df.index

                fig.add_trace(go.Candlestick(x=x_data, open=df['Open'], high=df['High'], 
                                             low=df['Low'], close=df['Close'], name='価格'), row=1, col=1)

                fig.add_trace(go.Scatter(x=x_data, y=df[f'SMA_{sma_short_window}'], 
                                         line=dict(color='blue', width=1.5), opacity=0.7, name=f'SMA {sma_short_window}'), row=1, col=1)
                fig.add_trace(go.Scatter(x=x_data, y=df[f'SMA_{sma_long_window}'], 
                                         line=dict(color='orange', width=1.5), opacity=0.7, name=f'SMA {sma_long_window}'), row=1, col=1)

                # 🌟 Prophet予測の描画 🌟
                if show_forecast and forecast is not None:
                    fig.add_trace(go.Scatter(
                        x=forecast['ds'], y=forecast['yhat_upper'],
                        mode='lines', line=dict(width=0), showlegend=False, name='予測上限'
                    ), row=1, col=1)

                    fig.add_trace(go.Scatter(
                        x=forecast['ds'], y=forecast['yhat_lower'],
                        mode='lines', line=dict(width=0), 
                        fill='tonexty', fillcolor='rgba(0, 176, 246, 0.2)', name='予測ブレ幅 (80%)'
                    ), row=1, col=1)

                    fig.add_trace(go.Scatter(
                        x=forecast['ds'], y=forecast['yhat'],
                        mode='lines', line=dict(color='rgba(0, 176, 246, 0.9)', width=2, dash='dot'), name='AI予測中央値'
                    ), row=1, col=1)

                # 【2段目】出来高
                colors = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red' for i in range(len(df))]
                fig.add_trace(go.Bar(x=x_data, y=df['Volume'], marker_color=colors, opacity=0.7, name='出来高'), row=2, col=1)

                # 【3段目】RSI
                fig.add_trace(go.Scatter(x=x_data, y=df['RSI'], line=dict(color='purple', width=1.5), name='RSI'), row=3, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

                # レイアウト設定
                fig.update_layout(height=900, xaxis_rangeslider_visible=False, margin=dict(l=50, r=50, t=50, b=50),
                                  title=f"{selected_ticker_name} テクニカル分析＆AI予測 ({selected_interval_label})")
                fig.update_yaxes(range=[0, 100], row=3, col=1)

                # 隙間を詰める処理 (休場日の非表示)
                if show_forecast and forecast is not None:
                    dt_all = pd.date_range(start=x_data[0], end=forecast['ds'].iloc[-1], freq=prophet_freq_mapping[interval])
                else:
                    dt_all = pd.date_range(start=x_data[0], end=x_data[-1], freq=prophet_freq_mapping[interval])
                
                # 存在するデータポイント（日付・時刻）の文字列セットを作成
                existing_dates_strs = set(x_data.strftime("%Y-%m-%d %H:%M:%S" if interval != "1d" else "%Y-%m-%d"))
                if show_forecast and forecast is not None:
                    existing_dates_strs.update(forecast['ds'].dt.strftime("%Y-%m-%d %H:%M:%S" if interval != "1d" else "%Y-%m-%d"))

                # 隙間の抽出
                dt_all_strs = dt_all.strftime("%Y-%m-%d %H:%M:%S" if interval != "1d" else "%Y-%m-%d").tolist()
                dt_breaks = [d for d in dt_all_strs if d not in existing_dates_strs]
                
                # Plotlyに隙間を無視させる
                if interval == "1d":
                    fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])
                else:
                    # 日中足の場合、休場時間（夜間など）や土日を一括で詰める
                    fig.update_xaxes(rangebreaks=[
                        dict(bounds=["sat", "mon"]), # 土日を除外
                    ])

                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"エラーが発生しました。詳細: {e}")
