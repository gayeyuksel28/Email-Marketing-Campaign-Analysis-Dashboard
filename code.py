import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

st.title("Email Marketing Campaign Analysis Dashboard")
st.write("This dashboard analyzes email marketing campaign data to predict customer engagement.")

# Load data function
def load_data():
    df = pd.read_csv('data.csv')
    df = df.dropna().reset_index(drop=True)
    df['Email Clicked'] = df['Email Clicked'].apply(lambda x: 0 if x == 0 else 1)
    return df
# Data Loading
st.header("📊 Data Overview")
st.write("Below is a preview of the first few rows of our dataset showing customer information and campaign metrics.")
df = load_data()
st.dataframe(df.head())

# Missing Data Analysis
st.header("🔍 Data Quality Check")
st.write("Checking for any missing values in the dataset to ensure data quality.")
missing_data = df.isnull().sum()
st.write(missing_data[missing_data > 0])

# Data Visualization
st.header("📈 Campaign Performance Visualization")

# Email Click Distribution
st.subheader("Email Click Distribution")
st.write("Distribution of emails clicked vs not clicked by customers.")
fig, ax = plt.subplots()
sns.countplot(x='Email Clicked', data=df, ax=ax)
ax.set_title("Email Clicked Distribution")
st.pyplot(fig)

# Correlation Matrix
st.subheader("Feature Correlation Analysis")
st.write("Heat map showing relationships between different numerical features.")
fig, ax = plt.subplots(figsize=(10, 6))
numeric_df = df.select_dtypes(include=[float, int])  # Select only numeric columns
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Data Preparation
categorical_features = ['Location', 'Gender']
numerical_features = ['Age', 'Email Opened', 'Product page visit', 'Discount offered']
X = df[['Age', 'Gender', 'Location', 'Email Opened', 'Product page visit', 'Discount offered']]
y = df['Email Clicked']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical_features),
    ]
)

X_transformed = preprocessor.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=34)

# Model Comparison
st.header("🤖 Model Performance")
st.write("Comparing different machine learning models for email click prediction.")
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(random_state=34)
}

model_scores = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    model_scores[model_name] = {'Accuracy': accuracy, 'AUC': auc}

scores_df = pd.DataFrame(model_scores).T
st.write(scores_df)

# Logistic Regression Model
best_model = models['Logistic Regression']
y_pred = best_model.predict(X_test)

# Confusion Matrix and Classification Report
st.subheader("Model Evaluation Metrics")
st.write("Detailed breakdown of model predictions showing true vs predicted values.")
st.write("### Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Clicked', 'Clicked'], yticklabels=['Not Clicked', 'Clicked'], ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
st.pyplot(fig)

st.write("### Classification Report")
st.text(classification_report(y_test, y_pred))

# Feature Importance
st.header("📊 Feature Impact Analysis")
st.write("Understanding which factors most strongly influence email click behavior.")
perm_importance = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=34)
importance_df = pd.DataFrame({
    'Feature': numerical_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)),
    'Importance': perm_importance.importances_mean
}).sort_values(by='Importance', ascending=False)
st.bar_chart(importance_df.set_index('Feature'))

# User Input Section
st.header("🎯 Click Probability Predictor")
st.write("Enter customer information below to predict the likelihood of email engagement.")
age = st.number_input("Age", min_value=0, max_value=100, value=30)
gender = st.selectbox("Gender", options=["Male", "Female"])
location = st.text_input("Location", value="New York")
email_opened = st.selectbox("Email Opened", options=[0, 1])
product_page_visit = st.number_input("Product Page Visit", min_value=0, value=1)
discount_offered = st.selectbox("Discount Offered", options=[0, 1])

if st.button("Predict"):
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [0 if gender == "Male" else 1],
        'Location': [location],
        'Email Opened': [email_opened],
        'Product page visit': [product_page_visit],
        'Discount offered': [discount_offered]
    })

    input_transformed = preprocessor.transform(input_data)
    prediction = best_model.predict(input_transformed)
    probability = best_model.predict_proba(input_transformed)[0, 1]
    if prediction[0] == 1:
        st.success(f"The customer is likely to click the email. Probability: {probability:.2f}")
    else:
        st.warning(f"The customer is not likely to click the email. Probability: {probability:.2f}")
