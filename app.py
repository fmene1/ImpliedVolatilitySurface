import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
import streamlit as st
import black_scholes_merton as bsm


st.set_page_config(layout="wide")

with st.sidebar as sidebar:
    st.title("Surface Parameters")
    form = st.form(key="Surface Parameters")
    ticker_name: str = form.text_input("Ticker", value="MSFT", help="Any ticker from Yahoo! Finance. For example try MSFT, AMZN, AAPL or GOOGL.")
    risk_free_rate: float = form.number_input("Risk Free Rate (%)", value=4, help = "Annualized risk free rate.") / 100
    div_yield: float = form.number_input("Dividend Yield (%)", value=1, help= "Annualized dividend yield.") / 100
    min_moneyness: float = form.number_input(
        "Minimum Moneyness", value=0.8, min_value=0.5, max_value=2.0, step=0.05, help = "Any option with moneyness less than this value will be dropped. Recommended value is 0.8."
    )
    max_moneyness: float = form.number_input(
        "Maximum Moneyness", value=1.2, min_value=0.5, max_value=2.0, step=0.05, help = "Any option with moneyness greater than this value will be dropped. Recommended value is 1.2."
    )
    form.form_submit_button("Generate Surface")

if min_moneyness >= max_moneyness:
    st.error(
        "Minimum Moneyness must be less than Maximum Moneyness. Please enter new values."
    )
    st.stop()

with st.status("Generating Surface...", expanded=True) as status:
    st.write("Fetching ticker data from Yahoo! Finance...")
    try:
        Ticker: yf.Ticker = yf.Ticker(ticker=ticker_name)
        last_price: pd.DataFrame = Ticker.history(period="1d")
        close_price: float = last_price["Close"].iloc[-1]
        close_date: pd.Timestamp = last_price.index[-1]
        st.write("Fetching option chain...")
        options_data: tuple[str] = Ticker.options
    except Exception as e:
        st.error("Something went wrong while fetching data. Please try again.")
        print(e)
        st.stop()
    expiration_dates: pd.DatetimeIndex = pd.to_datetime(options_data).tz_localize(
        close_date.tz
    )

    if len(expiration_dates) == 0:
        st.error("Something went wrong while fetching data. Please try again.")
        st.stop()

    calls_data = pd.DataFrame()
    for exp in expiration_dates:
        option_chain: pd.DataFrame = Ticker.option_chain(
            exp.strftime("%Y-%m-%d")
        ).calls[["strike", "bid", "ask", "lastPrice"]]
        # Drop data on options expiring in <7 days
        if days_to_exp := (exp - close_date).days < 7:
            continue
        option_chain["days_to_exp"] = (exp - close_date).days
        option_chain["mid"] = (option_chain["ask"] + option_chain["bid"]) / 2
        # Drop rows where mid price is <0
        option_chain = option_chain[option_chain["mid"] > 0]
        option_chain["market_price"] = option_chain["mid"]
        # option_chain["market_price"] = option_chain["lastPrice"]
        option_chain["moneyness"] = option_chain["strike"] / close_price
        # Drop rows where moneyness is outside the defined range
        option_chain = option_chain[
            (option_chain["moneyness"] <= max_moneyness)
            & (option_chain["moneyness"] >= min_moneyness)
        ]

        calls_data = pd.concat(
            [
                calls_data,
                option_chain[["strike", "moneyness", "days_to_exp", "market_price"]],
            ]
        )
    st.write("Calculating implied volatilities...")
    calls_data["implied_volatility_(%)"] = calls_data.apply(
        lambda x: bsm.implied_volatility(
            close_price,
            x["strike"],
            x["days_to_exp"] / 365,
            risk_free_rate,
            div_yield,
            x["market_price"],
        )
        * 100,
        axis=1,
    )

    calls_data.dropna(subset="implied_volatility_(%)", inplace=True)

    # calls_data.sort_values(by='days_to_exp', ascending = False, inplace = True)
    st.write("Creating graph...")
    x_data = (calls_data["days_to_exp"] / 365).values
    y_data = calls_data["moneyness"].values
    z_data = calls_data["implied_volatility_(%)"].values

    x = np.linspace(x_data.min(), x_data.max(), 50)
    # y = np.linspace(y_data.min(), y_data.max(), 50)
    y = np.linspace(min_moneyness, max_moneyness, 50)

    x_mesh, y_mesh = np.meshgrid(x, y)
    grid = griddata((x_data, y_data), z_data, (x_mesh, y_mesh), method="cubic")
    grid = np.ma.array(grid, mask=np.isnan(grid))

    fig: go.Figure = go.Figure(
        data=[
            go.Surface(
                x=x_mesh, y=y_mesh, z=grid, colorbar_title="Implied Volatility (%)"
            )
        ]
    )

    fig.update_layout(width=1000, height=800)

    fig.update_scenes(
        xaxis_title="Time to Expiration (years)",
        yaxis_title="Moneyness",
        zaxis_title="Implied Volatility (%)",
    )
    status.update(label="Surface Done!", state="complete", expanded=False)

st.title(f"Implied Volatility Surface for {Ticker.info['shortName']}")
st.plotly_chart(fig)