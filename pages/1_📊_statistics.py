import pandas as pd 
import streamlit as st

st.set_page_config(layout="wide")


@st.cache_data
def read_data():
    return pd.read_csv("data/satispay_reviews.csv")


data = read_data()

st.header("Data overview")

st.write(data)


stores = st.multiselect('Select stores',
                        ['google_play', 'app_store',],
                        ['google_play', 'app_store',])

resample = st.selectbox("select unit of time", ["daily", "weekly", "monthly", "yearly"])

resample_unit = resample[0]


plot_data = (data
            .assign(date=lambda x: pd.to_datetime(x.date))
            #.assign(score=lambda x: x.rating.map({"very negative": 1, "negative": 2, "neutral": 3, "positive": 4, "very positive": 5}))
            .query("store in @stores")
            .set_index("date")
            .resample(resample_unit)
            .rating
            .mean()
            .reset_index()
            )

st.header(f"Average {resample} rating for {stores}")
st.line_chart(plot_data, x="date", y="rating")

st.header("percentage of 1 star not adressed by devs")

plot_data = (data
            .assign(date=lambda x: pd.to_datetime(x.date))
            #.assign(score=lambda x: x.rating.map({"very negative": 1, "negative": 2, "neutral": 3, "positive": 4, "very positive": 5}))
            .query("store in @stores")
            .set_index("date")
            .query("rating==1")
            .assign(missing=lambda x: x.developerResponse.isna())
            .resample(resample_unit)
            .missing
            .mean()
            .reset_index()
            )

st.line_chart(plot_data, x="date", y="missing")