import streamlit as st
from PIL import Image
import pytesseract
import spacy
from transformers import pipeline

# Load SpaCy model for heading generation and BART for summarization
nlp = spacy.load("en_core_web_sm")

# Use the summarization model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", framework="pt")

# Function to generate the heading for the summarized text
def generate_heading(text):
    doc = nlp(text)
    nouns = [chunk.text for chunk in doc.noun_chunks]
    if nouns:
        return nouns[0].capitalize()
    return "Summary"

# Function to summarize the extracted OCR text without length limit
def summarize_text_with_heading(text):
    num_words = len(text.split())
    max_length = max(30, num_words // 3)  
    min_length = max(10, max_length // 2)
    
    # Summarize text
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    summary_text = summary[0]['summary_text']
    
    heading = generate_heading(text)
    
    return f"**{heading}**\n\n{summary_text}"

# Streamlit App Title
st.title("OCR and Text Summarization App")

# Initialize session state for OCR result if not already set
if 'ocr_result' not in st.session_state:
    st.session_state.ocr_result = ''

# Step 1: Image Upload for OCR
uploaded_file = st.file_uploader("Upload an image for OCR...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Button to process OCR
    if st.button("Extract Text using OCR"):
        with st.spinner("Extracting text..."):
            # Perform OCR on the uploaded image
            ocr_result = pytesseract.image_to_string(img)
            if ocr_result.strip():
                st.session_state.ocr_result = ocr_result  # Save OCR result in session state
            else:
                st.session_state.ocr_result = ''  # Clear OCR result if no text is detected
                st.error("No text detected in the image.")

# Display OCR result if it's already in session state and was successfully extracted
if st.session_state.ocr_result:
    st.markdown("### OCR Result:")
    st.write(st.session_state.ocr_result)

    # Step 2: Option to Summarize the Extracted Text
    if st.button("Summarize OCR Text"):
        with st.spinner("Summarizing text..."):
            summary = summarize_text_with_heading(st.session_state.ocr_result)
            st.markdown("### Summary:")
            st.write(summary)
