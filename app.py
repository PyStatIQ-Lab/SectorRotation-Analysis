import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from io import BytesIO
import pytz
import os
from pathlib import Path

# -------------------- Configuration --------------------
IST = pytz.timezone('Asia/Kolkata')
CACHE_EXPIRY_HOUR = 17  # 5:00 PM IST
CACHE_DIR = Path("./data_cache")
CACHE_FILE = CACHE_DIR / "sector_data.pkl"

# Create cache directory if it doesn't exist
CACHE_DIR.mkdir(exist_ok=True)

# Sector symbols mapping
sector_symbols = {
    "Financials": "^NSEBANK",
    "IT": "^CNXIT",
    "Auto": "^CNXAUTO",
    "FMCG": "^CNXFMCG",
    "Pharma": "^CNXPHARMA",
    "Metal": "^CNXMETAL",
    "Energy": "^CNXENERGY",
    "Infra": "^CNXINFRA",
    "Realty": "^CNXREALTY",
    "Media": "^CNXMEDIA"
}

# -------------------- Data Loading with Time-Based Caching --------------------
def should_fetch_fresh_data():
    """Check if it's time to fetch fresh data (after 5:00 PM IST)"""
    now = datetime.now(IST)
    return now.hour >= CACHE_EXPIRY_HOUR

def load_cached_data():
    """Load cached data if available and valid"""
    if CACHE_FILE.exists():
        cache_age = datetime.now() - datetime.fromtimestamp(CACHE_FILE.stat().st_mtime)
        if cache_age < timedelta(hours=24):
            try:
                return pd.read_pickle(CACHE_FILE)
            except:
                pass
    return None

def save_data_to_cache(data):
    """Save data to cache file"""
    try:
        data.to_pickle(CACHE_FILE)
        return True
    except:
        return False

def fetch_fresh_data(sectors):
    """Fetch fresh data from Yahoo Finance"""
    symbols = [sector_symbols[s] for s in sectors]
    try:
        # Always fetch 1 year of data ending at today's date
        end_date = datetime.now(IST)
        start_date = end_date - timedelta(days=365)
        
        data = yf.download(symbols, start=start_date, end=end_date)["Close"]
        data.columns = sectors  # Use friendly names
        save_data_to_cache(data)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def load_data(sectors):
    """Main data loading function with caching logic"""
    # First try to load cached data
    cached_data = load_cached_data()
    
    if cached_data is not None and not should_fetch_fresh_data():
        st.info("Using cached data from last fetch. Fresh data will be loaded after 5:00 PM IST.")
        return cached_data
    
    # If we need fresh data or cache is invalid
    with st.spinner("Fetching fresh market data..."):
        fresh_data = fetch_fresh_data(sectors)
        if fresh_data is not None:
            return fresh_data
        
        # If fresh fetch failed but we have cached data, use that with warning
        if cached_data is not None:
            st.warning("Using stale cached data as fresh fetch failed")
            return cached_data
        
        st.error("No data available. Please try again later.")
        return None

# -------------------- Analysis Functions --------------------
def calculate_returns(data):
    returns = pd.DataFrame(index=data.columns)
    for label, days in {"1M": 21, "3M": 63, "6M": 126, "1Y": 252}.items():
        if len(data) >= days:
            returns[label] = (data.iloc[-1] / data.iloc[-days] - 1) * 100
        else:
            returns[label] = np.nan
    return returns

def classify_sectors(df):
    result = {}
    for col in df.columns:
        sorted_sectors = df[col].sort_values(ascending=False)
        n = len(sorted_sectors)
        classification = pd.Series(index=sorted_sectors.index, dtype="object")
        classification[sorted_sectors.index[:n // 3]] = "Overweight"
        classification[sorted_sectors.index[n // 3:(n * 2) // 3]] = "Neutral"
        classification[sorted_sectors.index[(n * 2) // 3:]] = "Underweight"
        result[col] = classification
    return pd.DataFrame(result)

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def bollinger_bands(series, window=20, num_std=2):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return pd.DataFrame({
        'Middle Band': rolling_mean,
        'Upper Band': upper_band,
        'Lower Band': lower_band
    })

def compute_macd(series, slow=26, fast=12, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({
        'MACD': macd,
        'Signal': signal_line,
        'Histogram': macd - signal_line
    })

def volatility_analysis(df):
    returns = df.pct_change()
    return pd.DataFrame({
        'Annualized Volatility': returns.std() * np.sqrt(252),
        'Max Drawdown': (1 - df.div(df.cummax())).max()
    })

def capm_analysis(sector_returns):
    """Modified CAPM analysis with fixed dates"""
    market_returns = '^NSEI'
    risk_free = 0.05
    
    # Use same date range as sector data (last 1 year)
    end_date = datetime.now(IST)
    start_date = end_date - timedelta(days=365)
    
    market_data = yf.download(market_returns, start=start_date, end=end_date)['Close']
    market_returns = market_data.pct_change().dropna()
    
    results = {}
    for sector in sector_returns.columns:
        sector_ret = sector_returns[sector].dropna()
        merged = pd.concat([sector_ret, market_returns], axis=1).dropna()
        merged.columns = ['Sector', 'Market']
        
        covariance = merged.cov().iloc[0,1]
        market_variance = merged['Market'].var()
        beta = covariance / market_variance
        
        alpha = (merged['Sector'].mean() - risk_free/252) - beta * (merged['Market'].mean() - risk_free/252)
        
        results[sector] = {'Beta': beta, 'Alpha': alpha}
    
    return pd.DataFrame(results).T

def predict_returns(series, forecast_days=5):
    returns = series.pct_change().dropna()
    X = pd.DataFrame({
        'lag1': returns.shift(1),
        'lag2': returns.shift(2),
        'lag3': returns.shift(3)
    }).dropna()
    y = returns[X.index]
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    lr = LinearRegression().fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    
    last_features = np.array([returns.iloc[-1], returns.iloc[-2], returns.iloc[-3]]).reshape(1, -1)
    
    return {
        'Linear Regression': lr.predict(last_features)[0] * 100,
        'Random Forest': rf.predict(last_features)[0] * 100
    }

def sector_rotation_model(returns, volatility):
    scores = pd.DataFrame(index=returns.index)
    scores['Momentum'] = returns['3M'].rank(ascending=False)
    scores['Volatility'] = volatility['Annualized Volatility'].rank(ascending=True)
    scores['Composite'] = scores['Momentum'] * 0.7 + scores['Volatility'] * 0.3
    return scores.sort_values('Composite', ascending=False)

# -------------------- Streamlit App --------------------
def main():
    # Configure Streamlit page
    st.set_page_config(page_title="Sector Analysis Dashboard", layout="wide", page_icon="📊")
    
    # Sector selection only (no date selection)
    selected_sectors = st.sidebar.multiselect(
        "Select Sectors",
        list(sector_symbols.keys()),
        default=list(sector_symbols.keys())
    )
    
    # Load data (automatically handles 5PM IST refresh)
    data = load_data(selected_sectors)
    if data is None:
        st.stop()
    
    # Perform analysis
    with st.spinner("Performing analysis..."):
        returns = calculate_returns(data)
        sector_classification = classify_sectors(returns)
        rsi = data.apply(compute_rsi)
        rsi_latest = rsi.iloc[-1]
        momentum_scores = (returns["3M"] * 0.7 + rsi_latest * 0.3)
        correlation_matrix = data.pct_change().corr()
        
        # Technical indicators
        bb_signals = pd.Series(index=data.columns, dtype='object')
        for sector in data.columns:
            bb = bollinger_bands(data[sector])
            last_price = data[sector].iloc[-1]
            if last_price > bb['Upper Band'].iloc[-1]:
                bb_signals[sector] = "Overbought"
            elif last_price < bb['Lower Band'].iloc[-1]:
                bb_signals[sector] = "Oversold"
            else:
                bb_signals[sector] = "Neutral"
        
        macd_signals = pd.Series(index=data.columns, dtype='object')
        for sector in data.columns:
            macd = compute_macd(data[sector])
            if macd['MACD'].iloc[-1] > macd['Signal'].iloc[-1] and macd['MACD'].iloc[-2] <= macd['Signal'].iloc[-2]:
                macd_signals[sector] = "Bullish Crossover"
            elif macd['MACD'].iloc[-1] < macd['Signal'].iloc[-1] and macd['MACD'].iloc[-2] >= macd['Signal'].iloc[-2]:
                macd_signals[sector] = "Bearish Crossover"
            else:
                macd_signals[sector] = "Neutral"
        
        volatility = volatility_analysis(data)
        sector_returns = data.pct_change()
        capm_results = capm_analysis(sector_returns)
        
        ml_predictions = {}
        for sector in data.columns:
            ml_predictions[sector] = predict_returns(data[sector])
        
        rotation_recommendations = sector_rotation_model(returns, volatility)
    
    # Dashboard Layout
    st.title("Sector Analysis Dashboard")
    
    # Show last update time
    if CACHE_FILE.exists():
        last_update = datetime.fromtimestamp(CACHE_FILE.stat().st_mtime).astimezone(IST)
        st.markdown(f"""
        **Last Data Fetch:** {last_update.strftime('%Y-%m-%d %H:%M %Z')}  
        Next refresh after 5:00 PM IST
        """)
    
    # Summary Metrics
    st.header("📊 Performance Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sectors Analyzed", len(selected_sectors))
    with col2:
        st.metric("Latest Data", data.index[-1].strftime('%Y-%m-%d'))
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "📉 Technicals", "🤖 AI Forecast", "📋 Full Report"])
    
    with tab1:
        st.subheader("Return Analysis")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(returns, annot=True, cmap="RdYlGn", center=0, fmt=".1f", ax=ax)
        ax.set_title("Sector Returns (%)")
        st.pyplot(fig)
        
        st.subheader("Correlation Matrix")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
        ax.set_title("Sector Correlation Matrix")
        st.pyplot(fig)
        
        st.subheader("Momentum Analysis")
        momentum_df = pd.DataFrame({
            '3M Returns (%)': returns['3M'],
            'RSI': rsi_latest,
            'Momentum Score': momentum_scores,
            'Rating': sector_classification['3M']
        })
        st.dataframe(momentum_df.style.background_gradient(cmap='RdYlGn', axis=0))
    
    with tab2:
        st.subheader("Technical Indicators")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Bollinger Bands Signals**")
            st.dataframe(pd.DataFrame({
                'Price': data.iloc[-1].round(2),
                'Signal': bb_signals
            }).style.applymap(lambda x: 'background-color: #ffcccc' if x == 'Overbought' else 
                             ('background-color: #ccffcc' if x == 'Oversold' else ''), subset=['Signal']))
        
        with col2:
            st.markdown("**MACD Signals**")
            st.dataframe(pd.DataFrame({
                'MACD Signal': macd_signals
            }).style.applymap(lambda x: 'background-color: #ccffcc' if 'Bullish' in str(x) else 
                             ('background-color: #ffcccc' if 'Bearish' in str(x) else '')))
        
        st.subheader("Risk Analysis")
        risk_df = pd.DataFrame({
            'Annualized Volatility (%)': volatility['Annualized Volatility'] * 100,
            'Max Drawdown (%)': volatility['Max Drawdown'] * 100,
            'Beta': capm_results['Beta'],
            'Alpha': capm_results['Alpha']
        })
        st.dataframe(risk_df.style.background_gradient(cmap='YlOrRd', subset=['Annualized Volatility (%)', 'Max Drawdown (%)']))
    
    with tab3:
        st.subheader("Machine Learning Forecasts")
        
        forecast_df = pd.DataFrame({
            'Linear Regression (5D %)': [ml_predictions[s]['Linear Regression'] for s in data.columns],
            'Random Forest (5D %)': [ml_predictions[s]['Random Forest'] for s in data.columns]
        })
        st.dataframe(forecast_df.style.background_gradient(cmap='RdYlGn', axis=0))
        
        st.subheader("Sector Rotation Model")
        rotation_df = rotation_recommendations.copy()
        rotation_df['Rank'] = range(1, len(rotation_df) + 1)
        st.dataframe(rotation_df.style.background_gradient(cmap='RdYlGn', subset=['Composite']))
        
        st.markdown("""
        **Rotation Strategy Guide:**
        - **Top ranked sectors**: Best combination of momentum and low volatility
        - **Bottom ranked sectors**: Poor momentum or excessive volatility
        """)
    
    with tab4:
        st.subheader("Comprehensive Sector Report")
        
        # Create final output dataframe
        final_output = pd.DataFrame({
            '1M Return (%)': returns['1M'],
            '3M Return (%)': returns['3M'],
            '6M Return (%)': returns['6M'],
            '1Y Return (%)': returns['1Y'],
            'RSI': rsi_latest.round(1),
            'Momentum Score': momentum_scores.round(1),
            'Rating': sector_classification['3M'],
            'Bollinger Signal': bb_signals,
            'MACD Signal': macd_signals,
            'Volatility (%)': (volatility['Annualized Volatility'] * 100).round(1),
            'Max Drawdown (%)': (volatility['Max Drawdown'] * 100).round(1),
            'Beta': capm_results['Beta'].round(2),
            'Alpha': capm_results['Alpha'].round(4),
            'LR 5D Forecast (%)': [round(ml_predictions[s]['Linear Regression'], 1) for s in data.columns],
            'RF 5D Forecast (%)': [round(ml_predictions[s]['Random Forest'], 1) for s in data.columns],
            'Rotation Rank': rotation_recommendations['Composite'].rank().astype(int)
        })
        
        # Generate recommendations
        def get_recommendation(row):
            pct_rank = (final_output['Momentum Score'].rank(pct=True).loc[row.name])
            if pct_rank >= 0.75:
                return '🌟 Strong Buy'
            elif pct_rank >= 0.5:
                return '➖ Neutral'
            elif pct_rank >= 0.25:
                return '⚠️ Underweight'
            else:
                return '❌ Avoid'
        
        final_output['Recommendation'] = final_output.apply(get_recommendation, axis=1)
        
        st.dataframe(final_output.style
                    .background_gradient(cmap='RdYlGn', subset=['3M Return (%)', 'Momentum Score'])
                    .background_gradient(cmap='YlOrRd', subset=['Volatility (%)', 'Max Drawdown (%)'])
                    .applymap(lambda x: 'background-color: #ffcccc' if x == 'Overbought' else 
                             ('background-color: #ccffcc' if x == 'Oversold' else ''), subset=['Bollinger Signal'])
                    .applymap(lambda x: 'background-color: #ccffcc' if 'Bullish' in str(x) else 
                             ('background-color: #ffcccc' if 'Bearish' in str(x) else ''), subset=['MACD Signal'])
                    .applymap(lambda x: 'background-color: #e6ffe6' if '🌟' in str(x) else 
                             ('background-color: #ffe6e6' if '❌' in str(x) else ''), subset=['Recommendation']))
    
    # Sector Price Charts
    st.sidebar.markdown("---")
    selected_chart = st.sidebar.selectbox("View Price Chart", selected_sectors)
    
    if selected_chart:
        st.subheader(f"{selected_chart} Price Analysis")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Price and Bollinger Bands
        prices = data[selected_chart]
        bb = bollinger_bands(prices)
        prices.plot(ax=ax1, label='Price', color='blue')
        bb['Upper Band'].plot(ax=ax1, label='Upper Band', color='red', linestyle='--')
        bb['Middle Band'].plot(ax=ax1, label='Middle Band', color='green', linestyle='--')
        bb['Lower Band'].plot(ax=ax1, label='Lower Band', color='red', linestyle='--')
        ax1.set_title(f"{selected_chart} Price with Bollinger Bands")
        ax1.legend()
        ax1.grid(True)
        
        # MACD
        macd = compute_macd(prices)
        macd['MACD'].plot(ax=ax2, label='MACD', color='blue')
        macd['Signal'].plot(ax=ax2, label='Signal', color='orange')
        ax2.bar(macd.index, macd['Histogram'], 
                color=np.where(macd['Histogram'] > 0, 'green', 'red'),
                width=0.8, label='Histogram')
        ax2.axhline(0, color='gray', linestyle='--')
        ax2.set_title("MACD Indicator")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Download Report
    st.sidebar.markdown("---")
    if st.sidebar.button("📥 Download Full Report"):
        with st.spinner("Generating report..."):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                final_output.to_excel(writer, sheet_name='Summary')
                returns.to_excel(writer, sheet_name='Returns')
                correlation_matrix.to_excel(writer, sheet_name='Correlations')
                
                # Add technical indicators
                bb_data = pd.DataFrame({s: bollinger_bands(data[s]).iloc[-1] for s in data.columns}).T
                bb_data.to_excel(writer, sheet_name='Bollinger Bands')
                
                macd_data = pd.DataFrame({s: compute_macd(data[s]).iloc[-1] for s in data.columns}).T
                macd_data.to_excel(writer, sheet_name='MACD')
                
                # Add visualizations
                workbook = writer.book
                worksheet = writer.sheets['Summary']
                
                # Save plots to bytes
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                sns.heatmap(returns, annot=True, cmap="RdYlGn", center=0, fmt=".1f", ax=ax1)
                ax1.set_title("Sector Returns (%)")
                img1 = BytesIO()
                fig1.savefig(img1, format='png', bbox_inches='tight', dpi=300)
                plt.close(fig1)
                
                fig2, ax2 = plt.subplots(figsize=(10, 8))
                sns.heatmap(correlation_matrix, cmap="coolwarm", annot=True, fmt=".2f", ax=ax2)
                ax2.set_title("Sector Correlation Matrix")
                img2 = BytesIO()
                fig2.savefig(img2, format='png', bbox_inches='tight', dpi=300)
                plt.close(fig2)
                
                # Insert images
                worksheet.insert_image('K2', '', {'image_data': img1})
                worksheet.insert_image('K20', '', {'image_data': img2})
            
            output.seek(0)
            st.sidebar.download_button(
                label="⬇️ Download Excel Report",
                data=output,
                file_name=f"sector_analysis_{datetime.now(IST).strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    # About Section
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    **Sector Analysis Dashboard**  
    Version 1.0  
    Data provided by Yahoo Finance  
    Analysis performed on: {datetime.now(IST).strftime('%Y-%m-%d %H:%M %Z')}
    """)

if __name__ == "__main__":
    main()
