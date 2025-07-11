import streamlit as st
import pandas as pd
from fuzzywuzzy import fuzz
from transformers import pipeline
import re

# Initialize session state for conversation history
if 'history' not in st.session_state:
    st.session_state.history = []

# Load the Excel file with three sheets
@st.cache_data
def load_data():
    try:
        xls = pd.ExcelFile('kepler_data.xlsx')
        sheets = ['Draft', 'Admissions', 'Orientation', 'Programs']
        data = {}
        for sheet in sheets:
            df = pd.read_excel(xls, sheet_name=sheet)
            # Clean data: remove empty rows and ensure columns
            df = df.dropna(subset=['Questions', 'Answers'])
            df['Questions'] = df['Questions'].str.strip().replace('', None)
            df['Answers'] = df['Answers'].str.strip().replace('', None)
            data[sheet] = df[['Questions', 'Answers']].to_dict('records')
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return {}

# Initialize Hugging Face question-answering model
@st.cache_resource
def load_model():
    try:
        return pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to find the best matching question
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

# Main Streamlit app
def main():
    st.title("Kepler College Chatbot")
    st.write("Ask me anything about Kepler College, and I'll provide answers based on our dataset or generate a response if needed!")

    # Load data and model
    data = load_data()
    qa_model = load_model()

    # Input form
    with st.form(key='question_form', clear_on_submit=True):
        user_question = st.text_input("Your Question:", placeholder="e.g., What programs does Kepler College offer?")
        submit_button = st.form_submit_button("Ask")

    if submit_button and user_question:
        user_question = clean_input(user_question)
        if not user_question:
            st.warning("Please enter a valid question.")
            return

        # Add user question to history
        st.session_state.history.append({"role": "user", "message": user_question})

        # Find best match in dataset
        best_match, score = find_best_match(user_question, data)

        if best_match and score >= 80:
            answer, sheet = best_match
            response = f"{answer} (Source: {sheet} sheet)"
        else:
            # Fallback to Hugging Face model
            if qa_model:
                context = "Kepler College is a higher learning institution in Rwanda offering programs in Project Management, Business Analytics, and degrees through a partnership with Southern New Hampshire University (SNHU)."
                result = qa_model(question=user_question, context=context)
                response = result['answer']
                if result['score'] < 0.5:
                    response = f"Sorry, I couldn't find a close match in the dataset. Based on general knowledge: {response}"
            else:
                response = "Sorry, I couldn't find an answer in the dataset, and the AI model is unavailable."

        # Add bot response to history
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
