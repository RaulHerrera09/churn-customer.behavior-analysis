import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
import numpy as np

# Page configuration
st.set_page_config(page_title="Predictive Churn Model", layout="wide")

st.title("Predictive Churn Model")
st.markdown("""
This page demonstrates a complete Machine Learning pipeline to predict Churn.
We use a `RandomForestClassifier` model to identify patterns in customers who have left.
""")

# --- Data Loading ---


@st.cache_data
def load_data(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# --- Model Training Function ---
# We use cache_resource to save the trained model and pipeline


@st.cache_resource
def train_model(df):
    # 1. Define Features (X) and Target (y)
    # Exclude columns not relevant for prediction
    df_model = df.drop(
        columns=['RowNumber', 'CustomerId', 'Surname', 'Exited'])
    y = df['Exited']
    X = df_model

    # 2. Define Preprocessing
    # Identify numeric and categorical columns
    numeric_features = ['CreditScore', 'Age', 'Tenure',
                        'Balance', 'NumOfProducts', 'EstimatedSalary']
    categorical_features = ['Geography',
                            'Gender', 'HasCrCard', 'IsActiveMember']

    # Create transformer for numerical (scaling)
    numeric_transformer = StandardScaler()

    # Create transformer for categorical (One-Hot Encoding)
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Combine transformers with ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'  # Keep non-specified columns (if any)
    )

    # 3. Create the Pipeline

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
    ])

    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # 5. Train Model
    model.fit(X_train, y_train)

    # 6. Evaluate Model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # 7. Get Feature Importances
    # Get the feature names after OHE
    feature_names_out = model.named_steps['preprocessor'].get_feature_names_out(
    )
    importances = model.named_steps['classifier'].feature_importances_

    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names_out,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    return model, accuracy, precision, recall, cm, importance_df, X.columns


# --- Loading and Training ---
df_churn = load_data("data/Churn_modelling.csv")

if df_churn is not None:
    model, accuracy, precision, recall, cm, importance_df, feature_columns = train_model(
        df_churn)

    st.header("Model Results and Evaluation")
    st.markdown(
        "The model was trained on 80% of the data and evaluated on the remaining 20%.")

    # --- Evaluation Metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy:.2%}")
    col2.metric("Precision", f"{precision:.2%}",
                help="Of all customers the model *predicted* would churn, how many actually did?")
    col3.metric("Recall", f"{recall:.2%}",
                help="Of all customers who *actually* churned, how many did the model find?")

    st.markdown("""
    **Interpreting Metrics:**
    * **Accuracy:** What percentage of predictions were correct.
    * **Precision:** Important if the cost of contacting a customer is high. We don't want false positives.
    * **Recall:** **The most important metric here.** We want to *find* as many churning customers as possible, even if it means having some false positives. High Recall is key for a retention strategy.
    """)

    st.divider()

    # --- Evaluation Visualizations ---
    col1, col2 = st.columns(2)

    with col1:
        # Confusion Matrix Heatmap
        st.subheader("Confusion Matrix")

        # Invert 'cm' order for visualization (predicted on x, actual on y)
        z = cm
        x = ['Stays (Pred)', 'Exits (Pred)']
        y = ['Stays (Real)', 'Exits (Real)']

        fig_cm = ff.create_annotated_heatmap(
            z, x=x, y=y, colorscale='Blues', showscale=False)
        fig_cm.update_layout(title_text='Model Performance')
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        # Feature Importance
        st.subheader("Most Influential Features")
        fig_importance = px.bar(
            importance_df.head(10),
            x='Importance',
            y='Feature',
            orientation='h',
            title="Top 10 Features Predicting Churn"
        )
        fig_importance.update_layout(
            yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_importance, use_container_width=True)
        st.markdown("""
        **Analysis:** We see that `Age` ('num__Age'), `Balance` ('num__Balance'), and `NumOfProducts` are the strongest predictors.
        """)

    st.divider()

    # --- Interactive Simulator ---
    st.header("Churn Prediction Simulator")
    st.markdown(
        "Interact with the model to predict a customer's probability of churning.")

    # Create a form to group inputs
    with st.form(key='prediction_form'):
        st.subheader("Enter customer data:")

        col1, col2, col3 = st.columns(3)

        with col1:
            CreditScore = st.slider("Credit Score", 300, 850, 650)
            Age = st.slider("Age", 18, 100, 40)
            Tenure = st.slider("Tenure (years)", 0, 10, 5)

        with col2:
            Balance = st.number_input(
                "Account Balance", min_value=0.0, max_value=250000.0, value=10000.0, step=1000.0)
            NumOfProducts = st.slider("Number of Products", 1, 4, 1)
            Geography = st.selectbox("Country (Geography)", [
                                     'France', 'Spain', 'Germany'])

        with col3:
            Gender = st.selectbox("Gender", ['Male', 'Female'])
            HasCrCard = st.selectbox(
                "Has Credit Card", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
            IsActiveMember = st.selectbox(
                "Is Active Member", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
            EstimatedSalary = st.number_input(
                "Estimated Salary", min_value=0.0, max_value=200000.0, value=50000.0, step=1000.0)

        # Form submit button
        submit_button = st.form_submit_button(
            label='Predict Churn Probability')

    if submit_button:
        # Create a DataFrame with the input data
        input_data = {
            'CreditScore': [CreditScore],
            'Age': [Age],
            'Tenure': [Tenure],
            'Balance': [Balance],
            'NumOfProducts': [NumOfProducts],
            'EstimatedSalary': [EstimatedSalary],
            'Geography': [Geography],
            'Gender': [Gender],
            'HasCrCard': [HasCrCard],
            'IsActiveMember': [IsActiveMember]
        }

        # Ensure columns are in the correct order
        input_df = pd.DataFrame(input_data)[feature_columns]

        # Use the model to predict probability
        # model.predict_proba() returns [prob_no_churn, prob_churn]
        prediction_proba = model.predict_proba(input_df)[0]
        # Probability of Churn (Class 1)
        churn_probability = prediction_proba[1]

        # Show the result
        st.subheader("Prediction Result")
        if churn_probability > 0.5:
            st.error(
                f"High Risk! This customer has a **{churn_probability:.2%}** probability of churning.")
            st.markdown(
                "Recommending contacting this customer for a retention campaign.")
        else:
            st.success(
                f"Low Risk. This customer has a **{churn_probability:.2%}** probability of churning.")

else:
    st.warning("Could not load churn data for predictive analysis.")
