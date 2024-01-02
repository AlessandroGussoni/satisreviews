import pandas as pd 
import streamlit as st

st.set_page_config(layout="centered")

@st.cache_data
def read_data():
    return pd.read_csv("data/aspects.csv")


data = read_data()

st.header("Data overview")

st.write(data)

def render_aspect(aspect, size, value):
    min_val = 0
    current_val = round(value, 3)
    max_val = 5 

    progress_bar = st.empty()
    progress_text = st.empty()

    percentage = current_val / max_val
    
    progress_bar.progress(percentage)

    color = "green" if percentage * 100 > 2.5 else "red"
    
    progress_style = f"""

    <div style="
        display: flex;
        justify-content: space-between;
        align-items: center;"
        background-color: green;">
        <span>{aspect.upper()}, number of occurences {size} </span>
        <span></span>
        <span style="position: absolute; color: {color}; left: {percentage * 100}%; transform: translateX(-50%);">{current_val}</span>
        <span>5</span>
    </div>
    """
    progress_text.markdown(progress_style, unsafe_allow_html=True)

st.header("Aspect average score")


groups = (data
 .assign(score=lambda x: x.sentiment.map({"very negative": 1, "negative": 2, "neutral": 3, "positive": 4, "very positive": 5}))
 .groupby("group")
 .score
 .agg(["mean", "size"])
 .reset_index()
)

size_value = st.slider("aspect size cutoff", 2, 25, 20)
min_score, max_score = st.slider("score values range", 0.0, 5.0, (1.0, 4.0))

filtered_groups = (groups
                   .query("(size > @size_value)")
                   .query("(mean > @min_score) and (mean < @max_score)")
)


for g in filtered_groups.group.unique():
    render_aspect(g, 
                  filtered_groups.query("group == @g")["size"].values[0], 
                  filtered_groups.query("group == @g")["mean"].values[0])