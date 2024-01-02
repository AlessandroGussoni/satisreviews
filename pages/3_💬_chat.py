from openai import OpenAI
import streamlit as st
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import pandas as pd

import pinecone


@st.cache_data
def load_data():
    data = pd.read_csv("data/satispay_reviews.csv")
    data["id"] = data.index
    return data

@st.cache_resource
def load_index():
    pinecone.init(api_key="4ab9bc56-8709-428b-b79c-97e41fd37ef4", environment='gcp-starter')
    return pinecone.Index('satisreviews')

@st.cache_resource
def load_model():
    return SentenceTransformer('thenlper/gte-base')

data = load_data()
index = load_index()
model = load_model()


def cast_rag_prompt(model, index, prompt):
    vector = model.encode(prompt).astype(float)
    matches = index.query(vector=list(vector),
                          top_k=50)
    ids = [int(value["id"]) for value in matches["matches"]]
    reviews = data.query("id in @ids").review.tolist()
    reviews = [f"User_{i}: " + review for i, review in enumerate(reviews)]
    
    rag_prompt = f"""
    Look at the following reviews for a fintech mobile application:
    ---------------------
    {*reviews,}
    ---------------------
    Given the previous reviews and not prior knowledge, answer the following query.
    Query: {prompt}
    Answer: 
    """
    return rag_prompt


st.title("Chat with reviews")

client = OpenAI(api_key="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")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    if st.session_state.messages == []:
        rag_prompt = cast_rag_prompt(model, index, prompt)
        st.session_state.messages.append({"role": "user", "content": rag_prompt})
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            full_response += (response.choices[0].delta.content or "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
