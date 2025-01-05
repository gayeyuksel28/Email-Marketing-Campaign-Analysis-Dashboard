# Email-Marketing-Campaign-Analysis-Dashboard

# README: Email Campaign Prediction App

## Project Overview
This project aims to build a Streamlit-based web application that predicts the likelihood of a customer clicking an email in a marketing campaign. The application uses data preprocessing, visualization, machine learning models, and user inputs to provide insights and predictions.

## Features
1. **Data Loading and Preview**: Users can load and preview the dataset used in the analysis.
2. **Missing Data Analysis**: Displays missing values in the dataset to ensure data quality.
3. **Data Visualization**: Includes:
   - Email Clicked Distribution: A bar chart showing the number of emails clicked versus not clicked.
   - Correlation Matrix: A heatmap to understand relationships between numerical features.
4. **Model Comparison**: Evaluates Logistic Regression and Random Forest models based on accuracy and AUC scores.
5. **Model Evaluation**: Provides a confusion matrix and classification report for performance metrics.
6. **Feature Importance**: Visualizes the importance of features in predicting email clicks.
7. **User Input Section**: Allows users to input customer attributes and predict the likelihood of an email click.

## Visual Outputs
### 1. Email Clicked Distribution
![Email Clicked Distribution](path/to/your/image1.png)
This bar chart shows the distribution of clicked versus not clicked emails. It provides an overview of the class balance in the dataset, helping understand whether the data is imbalanced, which can impact model performance.

### 2. Correlation Matrix
![Correlation Matrix](path/to/your/image2.png)
The correlation matrix displays the relationships between numerical features in the dataset. Positive correlations are shown in red, and negative correlations in blue. For instance, features like "Discount Offered" and "Product Page Visit" exhibit a strong positive correlation, suggesting they might influence email clicks together.

### 3. Confusion Matrix
![Confusion Matrix](path/to/your/image3.png)
The confusion matrix visualizes the model's performance by displaying the number of true positives, true negatives, false positives, and false negatives. For example, the top-right cell indicates false positives (emails predicted as clicked but not clicked), while the bottom-left cell shows false negatives (emails not predicted as clicked but were actually clicked).

## Instructions to Run
1. Clone this repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Streamlit app with `streamlit run app.py`.
4. Upload your dataset if needed and interact with the app to visualize data and make predictions.

## Technical Details
- **Libraries Used**: Streamlit, Pandas, Scikit-learn, Seaborn, Matplotlib.
- **Models Implemented**:
  - Logistic Regression: Used for binary classification.
  - Random Forest: A robust ensemble model for classification tasks.

## How It Works
1. **Preprocessing**: Handles missing values, encodes categorical variables, and standardizes numerical features.
2. **Training and Testing**: Splits the dataset into training and testing sets (80/20 split).
3. **Model Comparison**: Trains Logistic Regression and Random Forest models and evaluates their performance.
4. **Prediction**: Uses user input and the trained model to predict the likelihood of a customer clicking an email.

## Future Improvements
- Add additional visualization options.
- Include more advanced machine learning models.
- Enhance user input flexibility with dynamic fields.

---
This README provides a comprehensive overview of the project, its features, and its visual outputs. Replace placeholder image paths (e.g., `path/to/your/image1.png`) with actual file paths or upload the images for accurate rendering.

