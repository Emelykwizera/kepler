import streamlit as st
import pandas as pd
from fuzzywuzzy import fuzz
import google.generativeai as genai
import os
import re

# Initialize session state for conversation history
if 'history' not in st.session_state:
    st.session_state.history = []

# Configure Gemini API (requires API key)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.error("Gemini API key not found. Please set GEMINI_API_KEY in Streamlit secrets.")
    st.stop()

# Load the Excel file with three sheets
@st.cache_data
def load_data():
    try:
        if not os.path.exists('kepler_data.xlsx'):
            st.error("Error: kepler_data.xlsx file not found in the app directory. Please ensure it's uploaded to the GitHub repository.")
            return {}
        
        xls = pd.ExcelFile('kepler_data.xlsx')
        sheets = ['Draft', 'Admissions', 'Orientation', 'Programs']
        data = {}
        for sheet in sheets:
            try:
                # Read first few rows to inspect headers
                df = pd.read_excel(xls, sheet_name=sheet, nrows=5)
                # Strip whitespace from column names
                df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
                actual_columns = df.columns.tolist()
                st.write(f"Debug: Columns in sheet '{sheet}' after stripping whitespace: {actual_columns}")
                
                # Check for required columns (case-insensitive, after stripping)
                columns_lower = [col.lower() for col in actual_columns]
                questions_col = next((col for col in actual_columns if col.lower().strip() == 'questions'), None)
                answers_col = next((col for col in actual_columns if col.lower().strip() == 'answers'), None)
                
                if not questions_col or not answers_col:
                    st.error(f"Error in sheet '{sheet}': Expected columns 'Questions' and 'Answers' (case-insensitive, no whitespace). Found: {actual_columns}")
                    continue
                
                # Read full sheet with correct header
                df = pd.read_excel(xls, sheet_name=sheet)
                # Strip whitespace from column names again
                df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
                # Rename columns to standard 'Questions' and 'Answers' if needed
                if questions_col != 'Questions':
                    df = df.rename(columns={questions_col: 'Questions'})
                if answers_col != 'Answers':
                    df = df.rename(columns={answers_col: 'Answers'})
                
                # Clean data
                df = df.dropna(subset=['Questions', 'Answers'])
                df['Questions'] = df['Questions'].astype(str).str.strip().replace('', None)
                df['Answers'] = df['Answers'].astype(str).str.strip().replace('', None)
                df = df.dropna(subset=['Questions', 'Answers'])
                
                if df.empty:
                    st.warning(f"Warning: No valid data in sheet '{sheet}' after cleaning.")
                    continue
                
                data[sheet] = df[['Questions', 'Answers']].to_dict('records')
            except Exception as e:
                st.error(f"Error processing sheet '{sheet}': {e}")
                continue
        
        if not data:
            st.error("Error: No valid data loaded from any sheet. Please check kepler_data.xlsx.")
        return data
    except Exception as e:
        st.error(f"Error loading kepler_data.xlsx: {e}")
        return {}

# Initialize Gemini model
@st.cache_resource
def load_model():
    try:
        return genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error(f"Error loading Gemini model: {e}")
        return None

# Find the best matching question
def find_best_match(question, data, threshold=80):
    best_match = None
    highest_score = 0
    for sheet, qa_pairs in data.items():
        for qa in qa_pairs:
            q = qa['Questions']
            if not isinstance(q, str):
                continue
            score = fuzz.token_sort_ratio(question.lower(), q.lower())
            if score > highest_score and score >= threshold:
                highest_score = score
                best_match = (qa['Answers'], sheet)
    return best_match, highest_score

# Clean user input
def clean_input(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

# Generate response using Gemini
def generate_gemini_response(model, question):
    try:
        context = "Kepler College is a higher learning institution in Rwanda offering programs in Project Management, Business Analytics, and degrees through a partnership with Southern New Hampshire University (SNHU)."
        prompt = f"Question: {question}\nContext: {context}\nAnswer based on the context or general knowledge about Kepler College, but keep it concise and relevant."
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {e}"

# Main Streamlit app
def main():
    st.title("Kepler College Chatbot")
    st.write("Ask me anything about Kepler College! I’ll answer based on our dataset or use Gemini AI for additional insights.")

    # Load data and model
    data = load_data()
    gemini_model = load_model()

    # Input form
    with st.form(key='question_form', clear_on_submit=True):
        user_question = st.text_input("Your Question:", placeholder="e.g., What programs does Kepler College offer?")
        submit_button = st.form_submit_button("Ask")

    if submit_button and user_question:
        user_question = clean
