# Export-Control-Class-Hierarchy

**Export-Control-Class-Hierarchy** is a Streamlit-based application designed to help users identify and categorize export control codes based on their input. This tool uses the `stsb-roberta-large` Sentence Transformer model to retrieve and rank relevant codes from a pre-processed dataset according to semantic similarity, with descriptions displayed in a tooltip-style interface for ease of use.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [License](#license)

## Project Overview

The **Export-Control-Class-Hierarchy** project enables users to input specific information related to export controls, returning the most relevant export control codes along with descriptions. This tool is ideal for individuals or organizations requiring quick access to categorized export information in an intuitive, user-friendly format.

The tool uses cosine similarity to match user input with pre-encoded export control descriptions, ensuring accurate and relevant code recommendations.

## Features

- **Semantic Similarity Matching**: Leverages the `stsb-roberta-large` model to generate embeddings and compute the similarity between user input and dataset entries.
- **Tooltip Interface**: Each code displays a tooltip with its description, which appears when hovering over the code.
- **Configurable Threshold and Results Limit**: Customize the relevance threshold and the maximum number of results displayed.
- **Streamlit UI**: Provides an interactive, web-based user interface built with Streamlit.

## Requirements

To run this project, you will need:
- **Python 3.7+**
- **Libraries**:
  - `Streamlit`
  - `Pandas`
  - `NumPy`
  - `SentenceTransformers`

Ensure these libraries are installed in your environment before running the project.

## Usage

To access the live application, visit: [Export Control Class Hierarchy](https://export-control-class-hierarchy.streamlit.app/)

### Instructions

1. **Enter Information**: In the text box provided, type any specific information related to export controls (e.g., "guns less than .5 caliber").
2. **Search**: Click the **Search** button to retrieve the most relevant export codes.
3. **Hover Over Codes**: Hover over each retrieved code to view a tooltip with its description.

## Project Structure

```plaintext
Export-Control-Class-Hierarchy/
├── Data_Dassault_Cleaned_with_Embeddings.pkl  # Precomputed embeddings and dataset file
├── app.py                                     # Main Streamlit application file
├── requirements.txt                           # Required libraries
└── README.md                                  # Project documentation
```
## Demo
https://export-control-class-hierarchy.streamlit.app/
