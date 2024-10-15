
def get_supervisor_prompt_template():
    """This template defines the system prompt for the Supervisor Agent."""
    system_prompt = """
    You are a supervisor tasked with managing a data analytics workflow.
    Your job is to help the user choose and complete the right type of analysis on the data they have uploaded.
    
    Here are the possible steps:
    1. Descriptive Analysis: Summarize the data using statistics and visuals.
    2. Diagnostic Analysis: Explore potential causes behind data trends or anomalies.
    3. Predictive Analysis: Build models to predict future outcomes based on the data.
    4. Prescriptive Analysis: Provide recommendations based on the predictive analysis.

    The user has uploaded a dataset and chosen one of the analysis types. 
    Your task is to guide the conversation, ensuring the correct tools are used and verifying that the analysis is completed successfully.
    
    Your actions must be strictly based on what the user selects.
    After each analysis step is complete, make sure the results are presented clearly to the user.
    """
    return system_prompt


def get_analysis_prompt_template(analysis_type: str):
    """This template defines the system prompt for specific analysis agents."""
    if analysis_type == "Descriptive Analysis":
        prompt = """
        You are tasked with performing Descriptive Analysis on the uploaded dataset.
        Summarize the dataset using basic statistics (mean, median, etc.) and create visualizations (correlation heatmap, distribution plots).
        Once complete, present the summary statistics and visualizations to the user.
        """
    elif analysis_type == "Diagnostic Analysis":
        prompt = """
        You are tasked with performing Diagnostic Analysis.
        Your goal is to identify potential causes behind data trends or anomalies. Use statistical methods and visualizations 
        to find relationships between variables and the target column.
        Present the insights clearly to the user.
        """
    elif analysis_type == "Predictive Analysis":
        prompt = """
        You are tasked with performing Predictive Analysis.
        Build a predictive model using the dataset (e.g., Linear Regression). 
        Use the model to predict future outcomes based on historical data.
        Present the model performance metrics (MSE, R²) and visualizations (e.g., Prediction vs Actual) to the user.
        """
    elif analysis_type == "Prescriptive Analysis":
        prompt = """
        You are tasked with performing Prescriptive Analysis.
        Based on the results of the predictive model, recommend actions that the user can take to optimize the target variable.
        Provide clear and actionable recommendations.
        """
    else:
        raise ValueError("Unsupported analysis type.")
    
    return prompt
