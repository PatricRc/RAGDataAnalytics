
# Data Analytics Assistant with Multi-Agent System

This project provides a **Streamlit-based Data Analytics Assistant** powered by a **multi-agent system**. The assistant helps non-technical users perform data analytics tasks on their datasets and even interact with the data through a **chatbot** interface using OpenAI's API.

## Features

- **Upload Dataset**: Users can upload their datasets in CSV or Excel format.
- **Data Analysis**: Supports various types of analysis such as:
  - **Descriptive Analysis**: Summarizes the dataset using basic statistics and visualizations.
  - **Diagnostic Analysis**: Identifies potential causes behind data trends or anomalies.
  - **Predictive Analysis**: Builds models to predict future outcomes based on historical data.
  - **Prescriptive Analysis**: Provides recommendations based on predictive model results.
  - **Detailed EDA**: Full Exploratory Data Analysis (EDA) to help users understand the data.
- **Chat with Data**: Users can ask questions about their dataset using OpenAI's GPT-4 model.
- **Visualizations**: The app automatically generates charts and visualizations to help users interpret the results of their analyses.

## Installation

To run this project locally, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/data-analytics-assistant.git
cd data-analytics-assistant
```

### 2. Install Dependencies

Make sure you have Python 3.8+ installed. Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Set Up OpenAI API Key

Create a `.env` file in the root directory (or use the provided template) and add your OpenAI API key:

```bash
# .env file
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Run the Application

Once everything is set up, run the Streamlit app:

```bash
streamlit run app.py
```

## Usage

1. **Upload Dataset**: Upload a CSV or Excel file containing your data.
2. **Choose Analysis Type**: Select from the available types of analysis or choose to chat with your data.
3. **View Results**: The app will display the analysis results and visualizations.
4. **Chat with Data**: If you selected the chat option, you can enter your query, and the app will use OpenAI's API to provide an answer based on the dataset.

## File Structure

- `app.py`: The main Streamlit application file.
- `data_loader.py`: Functions to load and clean datasets.
- `tools.py`: Contains the different types of analysis tools (descriptive, diagnostic, predictive, prescriptive, EDA).
- `agents.py`: Manages the multi-agent system and workflow.
- `prompts.py`: Defines the prompt templates for the agents.
- `custom_callback_handler.py`: Manages interactions between the agents and the Streamlit UI.
- `.env`: Stores the OpenAI API key.
- `requirements.txt`: Lists all the dependencies required for the project.

## Requirements

- Python 3.8+
- Streamlit
- OpenAI API Key

## License

This project is licensed under the MIT License. Feel free to contribute and make improvements!

