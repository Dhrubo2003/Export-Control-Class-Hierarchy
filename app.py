import streamlit as st
import pandas as pd
import numpy as np
import os
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
    # Explicitly convert vectors to NumPy arrays without additional arguments
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def desc(code):
    x = list(df[df['Full_Code'].str.startswith(code, na=False)]['Description'])
    return "\n".join(x)

# Function to find the most semantically relevant codes using NumPy for similarity
def find_relevant_codes_df(user_input, df, max_results=4, threshold=0.5):
    # Encode and normalize the user input to an embedding vector
    input_embedding = normalize_vector(model.encode(user_input))
    
    # Compute similarities between user input and dataset descriptions
    similarities = [cosine_similarity(input_embedding, normalize_vector(emb)) for emb in df['embedding']]
    similarities = np.array(similarities)
    
    # Sort by similarity scores in descending order
    top_results = np.argsort(-similarities)
    
    # Prepare the output, always include the top result
    relevant_codes = []
    relevant_codes.append((df.iloc[top_results[0]]["Full_Code"], float(similarities[top_results[0]]), desc(df.iloc[top_results[0]]["Full_Code"])))
    
    # Check the next results and only include if they meet the threshold
    for idx in top_results[1:max_results]:
        score = float(similarities[idx])
        if score >= threshold:
            relevant_codes.append((df.iloc[idx]["Full_Code"], score, desc(df.iloc[idx]["Full_Code"])))
        else:
            break  # Stop adding results if they don't meet the threshold
    
    return relevant_codes

# Streamlit UI for user input
user_input = st.text_input("Enter the information:")
button = st.button("Search")

tooltip_css = """
    <style>
    .tooltip-container {
        margin-bottom: 20px;  /* Add space between each code entry */
    }

    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
        margin-bottom: 10px;  /* Space between code and tooltip */
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 300px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 5px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        left: 0;  /* Align tooltip below the code */
        margin-top: 5px;  /* Space between the code and the description */
        opacity: 0;
        transition: opacity 0.3s;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
"""

# Inject custom CSS
st.markdown(tooltip_css, unsafe_allow_html=True)

if button:
    relevant_codes = find_relevant_codes_df(user_input, df, max_results=4, threshold=0.5)
    # Output the relevant codes and similarity scores
    for code, score, desc in relevant_codes:
        tooltip_html = f"""
        <div class="tooltip-container">
            <div class="tooltip">Code: USML.{code}
                <div class="tooltiptext">{desc}</div>
            </div>
        </div>
        """
        st.markdown(tooltip_html, unsafe_allow_html=True)
