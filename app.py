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
            st.error("Error: kepler_data.xlsx file not found in the app directory.")
            return {}
        
        xls = pd.ExcelFile('kepler_data.xlsx')
        sheets = ['Draft', 'Admissions', 'Orientation', 'Programs']
        data = {}
        for sheet in sheets:
            try:
                df = pd.read_excel(xls, sheet_name=sheet)
                # Check if required columns exist
                if 'Questions' not in df.columns or 'Answers' not in df.columns:
                    st.error(f"Error in sheet '{sheet}': Missing 'Questions' or 'Answers' column.")
                    continue
                # Clean data: remove empty rows and ensure valid strings
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
            st.error("Error: No valid data loaded from any sheet.")
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
        user_question = clean_input(user_question)
        if not user_question:
            st.warning("Please enter a valid question.")
            return

        st.session_state.history.append({"role": "user", "message": user_question})

        best_match, score = find_best_match(user_question, data)

        if best_match and score >= 80:
            answer, sheet = best_match
            response = f"{answer} (Source: {sheet} sheet)"
        else:
            if gemini_model:
                response = generate_gemini_response(gemini_model, user_question)
                response = f"Sorry, I couldn't find a close match in the dataset. Based on Gemini AI: {response}"
            else:
                response = "Sorry, I couldn't find an answer in the dataset, and the Gemini model is unavailable."

        st.session_state.history.append({"role": "bot", "message": response})

    # Display conversation history
    st.subheader("Conversation History")
    for entry in st.session_state.history:
        if entry['role'] == 'user':
            st.markdown(f"**You**: {entry['message']}")
        else:
            st.markdown(f"**KeplerBot**: {entry['message']}")

if __name__ == "__main__":
    main()
