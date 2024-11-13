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

st.markdown(
    """
    <style>
    .main-content {
        padding: 20px;
        border-radius: 8px;
        background-color: #f9f9f9;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .tooltip-container {
        margin-bottom: 20px;
    }
    
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
        font-size: 1.1em;
        font-weight: bold;
        color: #2c3e50;
        padding: 10px;
        background-color: #ecf0f1;
        border-radius: 5px;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 300px;
        background-color: #555;
        color: #fff;
        text-align: left;
        border-radius: 5px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        left: 0;
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

user_input = st.text_input("Enter your search information:")
button = st.button("Search")

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
