import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Tuple
from pydantic import BaseModel
from openai import OpenAI
import os
import json
import io
import base64

# Load environment variables
# load_dotenv()

# Initialize support flags
PDF_SUPPORT = False
DOCX_SUPPORT = False

# Try to import PyPDF2 and docx
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    st.warning("PyPDF2 library is not installed. PDF support is disabled. To enable PDF support, run: pip install PyPDF2")

try:
    import docx
    import docx2txt
    DOCX_SUPPORT = True
except ImportError:
    st.warning("python-docx and docx2txt libraries are not installed. DOCX support is disabled. To enable DOCX support, run: pip install python-docx docx2txt")

class DocumentClassification(BaseModel):
    classification: str

class ExtractedData(BaseModel):
    data: Dict[str, Any]


##

def read_pdf(file):
    if not PDF_SUPPORT:
        st.error("PDF support is not available. Please install PyPDF2.")
        return None
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def read_docx(file):
    if not DOCX_SUPPORT:
        st.error("DOCX support is not available. Please install python-docx and docx2txt.")
        return None
    text = docx2txt.process(file)
    return text

def classify_document(file_content: str) -> DocumentClassification:
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a document classification expert. Classify the following document into one of these categories: Invoice, Receipt, Contract, Order Confirmation, Order, Insurance Case, HR Document or Report. Respond with a JSON object containing a 'classification' key."},
                {"role": "user", "content": f"Classify this document and respond with a JSON object:\n\n{file_content}"}
            ],
            response_format={"type": "json_object"}
        )
        response_content = json.loads(completion.choices[0].message.content)
        return DocumentClassification(**response_content)
    except Exception as e:
        st.error(f"Error in classify_document: {str(e)}")
        raise

def extract_data(file_content: str, doc_type: str) -> ExtractedData:
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are a data extraction expert. Extract key information from this {doc_type} and return it as a JSON object."},
                {"role": "user", "content": f"Extract key information from this {doc_type} and respond with a JSON object:\n\n{file_content}"}
            ],
            response_format={"type": "json_object"}
        )
        response_content = json.loads(completion.choices[0].message.content)
        return ExtractedData(data=response_content)
    except Exception as e:
        st.error(f"Error in extract_data: {str(e)}")
        raise

def display_pdf(file):
    try:
        # Streamlit supports directly displaying PDFs using st.file_uploader
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64.b64encode(file.read()).decode()}" width="700" height="1000" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying PDF: {str(e)}")

#def display_pdf(file):
    #base64_pdf = base64.b64encode(file.getvalue()).decode('utf-8')
    #pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    #st.markdown(pdf_display, unsafe_allow_html=True)

def chat_with_ai_stream(file_content: str, user_message: str, chat_history: List[Tuple[str, str]]):
    try:
        messages = [
            {"role": "system", "content": "You are an AI assistant helping to analyze and discuss document content."},
            {"role": "user", "content": f"Here's the document content:\n\n{file_content}"}
        ]
        for user_msg, ai_msg in chat_history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": ai_msg})
        messages.append({"role": "user", "content": user_message})
        
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True
        )
        return stream
    except Exception as e:
        st.error(f"Error in chat_with_ai_stream: {str(e)}")
        raise

# Existing functions (read_pdf, read_docx, classify_document, extract_data, display_pdf, chat_with_ai_stream) remain unchanged

# Streamlit app
st.title("Document Classifier and Data Extractor")

# API Key input
api_key = st.text_input("Enter your OpenAI API Key", type="password")

if api_key:
    # Set up OpenAI client
    client = OpenAI(api_key=api_key)

    # Rest of the app
    file_types = ["txt"]
    if PDF_SUPPORT:
        file_types.append("pdf")
    if DOCX_SUPPORT:
        file_types.extend(["doc", "docx"])

    uploaded_file = st.file_uploader("Choose a file", type=file_types)

    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        
        try:
            # Read file content based on file type
            if uploaded_file.type == "text/plain":
                file_content = uploaded_file.getvalue().decode("utf-8")
                st.subheader("Document Content:")
                st.text_area("", file_content, height=300)
            elif uploaded_file.type == "application/pdf" and PDF_SUPPORT:
                file_content = read_pdf(uploaded_file)
                st.subheader("Document Content (PDF):")
                display_pdf(uploaded_file)
            elif uploaded_file.type in ["application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"] and DOCX_SUPPORT:
                file_content = read_docx(uploaded_file)
                st.subheader("Document Content:")
                st.text_area("", file_content, height=300)
            else:
                st.error("Unsupported file type or required library not installed")
                st.stop()

            if file_content is None:
                st.stop()

            # Classify the document
            classification_result = classify_document(file_content)
            doc_type = classification_result.classification
            st.write(f"Document classified as: {doc_type}")
            
            # Extract data
            extraction_result = extract_data(file_content, doc_type)
            
            # Display extracted data
            st.subheader("Extracted Data (JSON):")
            st.json(extraction_result.data)
            
            # Display the extracted data in a table format
            st.subheader("Extracted Data (Table):")
            df = pd.DataFrame([extraction_result.data])
            st.table(df)

            # Download buttons for JSON output
            st.subheader("Download JSON Output")
            
            # Classification JSON
            st.download_button(
                label="Download Classification JSON",
                data=classification_result.model_dump_json(indent=2),
                file_name="classification.json",
                mime="application/json"
            )
            
            # Extraction JSON
            st.download_button(
                label="Download Extracted Data JSON",
                data=extraction_result.model_dump_json(indent=2),
                file_name="extracted_data.json",
                mime="application/json"
            )

            # Chat window with streaming and history
            st.subheader("Chat about the Document")
            
            # Initialize chat history
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []

            # Display chat history
            for user_msg, ai_msg in st.session_state.chat_history:
                st.text_area("User:", value=user_msg, height=100, disabled=True)
                st.text_area("AI:", value=ai_msg, height=100, disabled=True)
                st.markdown("---")

            user_message = st.text_input("Ask a question about the document:")
            if user_message:
                st.text_area("User:", value=user_message, height=100, disabled=True)
                stream = chat_with_ai_stream(file_content, user_message, st.session_state.chat_history)
                response_placeholder = st.empty()
                full_response = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        response_placeholder.text_area("AI:", value=full_response + "▌", height=100, disabled=True)
                response_placeholder.text_area("AI:", value=full_response, height=100, disabled=True)
                
                # Add the new message pair to chat history
                st.session_state.chat_history.append((user_message, full_response))
                st.markdown("---")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    st.sidebar.header("About")
    st.sidebar.info("This app uses OpenAI's GPT model to classify documents, extract relevant information in a structured format, and chat about the document content with real-time streaming responses. The chat history is maintained for context. Upload a TXT file (or PDF/DOC/DOCX if supported) to see it in action!")

    # Display information about supported file types
    supported_types = "TXT"
    if PDF_SUPPORT:
        supported_types += ", PDF"
    if DOCX_SUPPORT:
        supported_types += ", DOC, DOCX"
    st.sidebar.info(f"Supported file types: {supported_types}")

    if not (PDF_SUPPORT and DOCX_SUPPORT):
        st.sidebar.warning("To enable full PDF and DOCX support, run: pip install PyPDF2 python-docx docx2txt")

else:
    st.warning("Please enter your OpenAI API Key to use the application.")