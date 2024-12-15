import streamlit as st
from scrapee import (
    scrape_market_data,
    # scrape_social_media
)
import pandas as pd

st.title("AI Memecoin Tracker")

st.subheader("Step 1: Fetch Market Data")
if st.button("Fetch Market Data"):
    st.write("Fetching low market cap coins from CoinGecko...")
    try:
        market_data = scrape_market_data()
        
        st.session_state.market_data = market_data
        st.success("Market data fetched successfully!")

        st.dataframe(pd.DataFrame(market_data))
    except Exception as e:
        st.error(f"Error fetching market data: {e}")

# if "market_data" in st.session_state:
#     st.subheader("Step 2: Analyze Social Media Mentions")
#     coin = st.selectbox("Select a coin for social media analysis", [c["symbol"] for c in st.session_state.market_data])

#     if st.button("Analyze Social Media"):
#         st.write(f"Fetching Twitter mentions for {coin}...")
#         try:
#             mentions = scrape_social_media(coin)
#             st.success(f"Found {len(mentions)} mentions for {coin}.")
#             st.write(mentions)
#         except Exception as e:
#             st.error(f"Error analyzing social media: {e}")