import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# File path for saving embeddings
embedding_file = 'Data_Dassault_Cleaned_with_Embeddings.pkl'

model = SentenceTransformer('stsb-roberta-large')
df = pd.read_pickle(embedding_file)

# Function to normalize a vector using NumPy
def normalize_vector(vec):
    return vec / np.linalg.norm(vec)

# Function to compute cosine similarity using normalized vectors
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def desc(code):
    x = list(df[df['Full_Code'].str.startswith(code, na=False)]['Description'])
    return "\n".join(x)

# Function to find the most semantically relevant codes using cosine similarity
def find_relevant_codes_df(user_input, df, max_results=4, threshold=0.5):
    input_embedding = normalize_vector(model.encode(user_input))
    similarities = [cosine_similarity(input_embedding, normalize_vector(emb)) for emb in df['embedding']]
    similarities = np.array(similarities)
    top_results = np.argsort(-similarities)
    
    relevant_codes = []
    relevant_codes.append((df.iloc[top_results[0]]["Full_Code"], float(similarities[top_results[0]]), desc(df.iloc[top_results[0]]["Full_Code"])))
    
    for idx in top_results[1:max_results]:
        score = float(similarities[idx])
        if score >= threshold:
            relevant_codes.append((df.iloc[idx]["Full_Code"], score, desc(df.iloc[idx]["Full_Code"])))
        else:
            break
    
    return relevant_codes

# Streamlit UI styling
st.set_page_config(page_title="Export Control Class Hierarchy", layout="centered")
st.title("🔍 Export Control Class Hierarchy")

# Background and main content styling
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to bottom right, #4A00E0, #8E2DE2);  /* Gradient background */
        color: #FFFFFF;
    }
    .main-content {
        max-width: 600px;
        margin: auto;
        padding: 20px;
        border-radius: 10px;
        background-color: rgba(255, 255, 255, 0.1);
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
    }
    .stTextInput>div>div>input {
        width: 100% !important;
        padding: 12px;
        border-radius: 8px;
        border: none;
        margin-bottom: 15px;
    }
    .stButton>button {
        width: 100%;
        background-color: #6A1B9A;
        color: #FFFFFF;
        padding: 10px;
        border: none;
        border-radius: 8px;
        font-size: 16px;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #8E44AD;
    }
    .tooltip-container {
        margin-top: 20px;
        font-size: 1.1em;
        color: #FFFFFF;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        font-weight: bold;
        cursor: pointer;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 350px;
        background-color: #2E2E2E;
        color: #FFF;
        text-align: left;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        left: 50%;
        margin-left: -175px;
        margin-top: 5px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-content">', unsafe_allow_html=True)

# User input box and button
user_input = st.text_input("Enter your search information:")
button = st.button("Search")

# Display relevant codes when button is clicked
if button:
    relevant_codes = find_relevant_codes_df(user_input, df, max_results=4, threshold=0.5)
    
    if relevant_codes:
        for code, score, desc in relevant_codes:
            tooltip_html = f"""
            <div class="tooltip-container">
                <div class="tooltip">Code: USML.{code}
                    <div class="tooltiptext">{desc}</div>
                </div>
            </div>
            """
            st.markdown(tooltip_html, unsafe_allow_html=True)
    else:
        st.write("No relevant codes found. Try adjusting your search terms.")

st.markdown('</div>', unsafe_allow_html=True)
