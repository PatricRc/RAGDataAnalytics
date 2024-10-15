
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from custom_callback_handler import CustomStreamlitCallbackHandler
from agents import define_graph
from data_loader import load_data_file, clean_data
from tools import get_analysis_tool
from langchain.chat_models import ChatOpenAI

# Load environment variables
load_dotenv()

# Streamlit Page Setup
st.set_page_config(layout="wide", page_title="Data Analysis Assistant")

# Supervisor Agent Setup
st.title("Data Analytics Assistant 🧠")
st.markdown("This app helps non-technical users build and manage data analytics projects with ease.")

# Sidebar: File Upload Section
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

# Sidebar: Select Analysis Type
analysis_types = ["Descriptive Analysis", "Diagnostic Analysis", "Predictive Analysis", "Prescriptive Analysis", "Detailed EDA", "Chat with Data"]
selected_analysis = st.sidebar.selectbox("Select the type of analysis", analysis_types)

# Supervisor Agent Initialization
flow_graph = define_graph()
callback_handler = CustomStreamlitCallbackHandler(st.container())

@st.cache_data
def load_data(uploaded_file):
    """Load data from the uploaded file."""
    try:
        # Load based on file type
        if uploaded_file.name.endswith("xlsx"):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        elif uploaded_file.name.endswith("csv"):
            df = pd.read_csv(uploaded_file)
        else:
            st.error("Unsupported file type.")
            return None

        return df

    except Exception as e:
        st.error(f"Error processing the file: {e}")
        return None

def chat_with_data(df_chat, input_text, api_key):
    """Chat with the uploaded dataset using OpenAI's API."""
    try:
        # Convert DataFrame to a format suitable for context
        context = df_chat.to_string(index=False)

        # Create a prompt template for OpenAI
        message = f"""
        Answer the following question using the context provided:

        Context:
        {context}

        Question:
        {input_text}

        Answer:
        """

        # Initialize OpenAI with GPT-4 model
        llm = ChatOpenAI(model_name="gpt-4", openai_api_key=api_key)

        # Generate the response from OpenAI
        response = llm.predict(message)

        st.write(response)

    except Exception as e:
        st.error(f"Error during chat: {e}")

# Main logic
if uploaded_file:
    # Load and clean the data
    file_path = os.path.join("temp", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load data
    data = load_data(uploaded_file)

    if data is not None:
        st.write("### Uploaded Data (First 10 Rows)")
        st.dataframe(data.head(10))  # Display the first few rows of the dataset

        # Option: Start Analysis or Chat with Data
        if selected_analysis != "Chat with Data":
            if st.button("Start Analysis"):
                st.write(f"Starting {selected_analysis}...")

                # Use the appropriate analysis tool
                analysis_tool = get_analysis_tool(selected_analysis)
                supervisor_output = analysis_tool(data)

                # Display analysis results
                st.write("### Analysis Results")

                if isinstance(supervisor_output, str):
                    st.write(supervisor_output)
                elif isinstance(supervisor_output, dict):
                    st.write(f"Mean Squared Error (MSE): {supervisor_output['mse']}")
                    st.write(f"R² Score: {supervisor_output['r2']}")
                else:
                    st.dataframe(supervisor_output)

                # Display visualizations (images saved during analysis)
                st.write("### Visualizations")
                # Add code to display images based on the analysis type...
        else:
            # Chat with Data Option
            st.write("### Chat with Your Data")

            # Prompt for OpenAI API Key
            api_key = st.text_input("Enter your OpenAI API Key", type="password")

            # Enter the question for OpenAI
            input_text = st.text_area("Ask a question about your data")

            # Chat with the data when both inputs are provided
            if input_text and api_key and st.button("Chat with Data"):
                chat_with_data(data, input_text, api_key)
else:
    st.write("Please upload a CSV or Excel file to proceed.")
