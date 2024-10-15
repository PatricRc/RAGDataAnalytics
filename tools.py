
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import missingno as msno


def descriptive_analysis(data: pd.DataFrame):
    """Perform Descriptive Analysis by summarizing the dataset with statistics and visualizations."""
    summary_stats = data.describe()

    # Correlation Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.savefig('temp/correlation_heatmap.png')
    plt.close()

    # Histograms and Box Plots for Numerical Columns
    numeric_columns = data.select_dtypes(include=['float', 'int']).columns
    for column in numeric_columns:
        plt.figure(figsize=(10, 5))

        # Subplot 1: Histogram
        plt.subplot(1, 2, 1)
        sns.histplot(data[column], kde=True, bins=30, color='blue')
        plt.axvline(data[column].mean(), color='red', linestyle='--', linewidth=2)
        plt.title(f'Distribution of {column}')

        # Subplot 2: Box Plot
        plt.subplot(1, 2, 2)
        sns.boxplot(data=data, y=column, color='orange')
        plt.title(f'Box Plot of {column}')

        plt.tight_layout()
        plt.savefig(f'temp/{column}_distribution_boxplot.png')
        plt.close()

    return summary_stats


def diagnostic_analysis(data: pd.DataFrame, target_column: str):
    """Perform Diagnostic Analysis to identify possible causes behind trends or anomalies."""
    sns.pairplot(data, hue=target_column)
    plt.savefig('temp/pairplot.png')
    plt.close()

    correlation_with_target = data.corr()[target_column].sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.kdeplot(data[target_column], shade=True, color="g")
    plt.title(f'KDE Plot for {target_column}')
    plt.savefig('temp/kdeplot_target.png')
    plt.close()

    return correlation_with_target


def predictive_analysis(data: pd.DataFrame, target_column: str):
    """Perform Predictive Analysis by building a simple Linear Regression model."""
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Prediction vs Actual')
    plt.savefig('temp/prediction_vs_actual.png')
    plt.close()

    importance = pd.Series(model.coef_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance.values, y=importance.index)
    plt.title("Feature Importance in Linear Regression")
    plt.savefig('temp/feature_importance.png')
    plt.close()

    return {'mse': mse, 'r2': r2}


def prescriptive_analysis(data: pd.DataFrame, target_column: str):
    """Perform Prescriptive Analysis to recommend actions based on predictive models."""
    correlation_with_target = data.corr()[target_column].sort_values(ascending=False)
    most_important_feature = correlation_with_target.index[1]

    recommendation = f"Improve {most_important_feature} to optimize {target_column}."

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data[most_important_feature], y=data[target_column])
    plt.title(f'{most_important_feature} vs {target_column}')
    plt.savefig(f'temp/{most_important_feature}_vs_{target_column}.png')
    plt.close()

    return recommendation


def detailed_eda(data: pd.DataFrame):
    """Perform a detailed Exploratory Data Analysis (EDA) on the dataset."""
    st.write("## General Dataset Overview")
    st.write("### Data Types and Missing Values")
    st.dataframe(data.info())

    st.write("### Summary Statistics for Numerical Columns")
    st.dataframe(data.describe())

    st.write("### Summary Statistics for Categorical Columns")
    st.dataframe(data.describe(include=['object', 'category']))

    st.write("## Missing Data Overview")
    plt.figure(figsize=(10, 6))
    msno.matrix(data)
    plt.title('Missing Data Matrix')
    plt.savefig('temp/missing_data_matrix.png')
    plt.close()

    st.write("## Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.savefig('temp/correlation_heatmap_eda.png')
    plt.close()

    numeric_columns = data.select_dtypes(include=['float', 'int']).columns
    st.write("## Distribution and Box Plot for Numerical Columns")
    for column in numeric_columns:
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        sns.histplot(data[column], kde=True, bins=30, color='blue')
        plt.axvline(data[column].mean(), color='red', linestyle='--', linewidth=2)
        plt.title(f'Distribution of {column}')

        plt.subplot(1, 2, 2)
        sns.boxplot(data=data, y=column, color='orange')
        plt.title(f'Box Plot of {column}')

        plt.tight_layout()
        plt.savefig(f'temp/{column}_distribution_boxplot_eda.png')
        plt.close()

    categorical_columns = data.select_dtypes(include=['object', 'category']).columns
    for column in categorical_columns:
        if data[column].nunique() <= 10:
            plt.figure(figsize=(6, 6))
            data[column].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette('tab10'))
            plt.title(f'Pie Chart of {column}')
            plt.axis('equal')
            plt.savefig(f'temp/{column}_piechart_eda.png')
            plt.close()

    return "EDA completed."


def get_analysis_tool(analysis_type: str):
    if analysis_type == "Descriptive Analysis":
        return descriptive_analysis
    elif analysis_type == "Diagnostic Analysis":
        return diagnostic_analysis
    elif analysis_type == "Predictive Analysis":
        return predictive_analysis
    elif analysis_type == "Prescriptive Analysis":
        return prescriptive_analysis
    elif analysis_type == "Detailed EDA":
        return detailed_eda
    else:
        raise ValueError("Unsupported analysis type.")
