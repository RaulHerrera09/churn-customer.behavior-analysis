# churn-customer.behavior-analysis

# Bank Simulation Dashboard 

Welcome to my project! This is an interactive dashboard I built with Streamlit for my Data Analyst portfolio. The goal was to simulate the type of analysis that would be performed at a digital bank, covering the entire workflow: from exploratory data analysis (EDA) to a predictive Machine Learning model.

# Key Features
This dashboard is a multi-page application that allows you to navigate between different analyses:

1. General Overview: A high-level demographic profile of the customer base (using Bank_marketing.csv).

2. Churn Analysis: A deep dive into why customers leave the bank, identifying key factors (using Churn_modelling.csv).

3. Loan Insights: A profiling analysis to discover the characteristics of the "ideal customer" who accepts a personal loan (using Bank_Loan_Modelling - Data.csv).

4. Predictive Model: The most exciting part! A Machine Learning model (Random Forest) trained to predict the probability of a customer churning. It includes an interactive simulator where you can input a customer's data and get a real-time prediction.

# Tech Stack
For this project, I focused on using industry-standard tools for data analysis and prototyping:

Data Analysis: pandas and numpy.

Data Visualization: plotly (for the interactive charts), seaborn, and matplotlib.

Machine Learning: scikit-learn (I specifically used Pipeline, StandardScaler, OneHotEncoder, RandomForestClassifier, and LogisticRegression for comparison).

Web Application (Dashboard): streamlit.

Development Environment: jupyter (for the exploratory notebook) and vscode.

# Project Structure
The project is organized as follows to separate the application logic, data, and exploratory analysis.

.
├── app/
│   ├── app.py              # Main page (Home)
│   └── pages/              # Dashboard pages
│       ├── 1_Overview.py
│       ├── 2_Churn_Analysis.py
│       ├── 3_Loan_Insights.py
│       └── 4_Predictive_Model.py
├── data/                   # CSV data files
│   ├── Bank_Loan_Modelling - Data.csv
│   ├── Churn_modelling.csv
│   └── Bank_marketing.csv
├── notebooks/
│   └── Master_Analysis_Notebook.ipynb  # My analysis & modeling "scratchpad"
├── .gitignore
├── README.md               # What you're reading!
└── requirements.txt        # The required libraries

# How to Run Locally
If you want to try the application on your machine, just follow these steps:

1. Clone this repository:

Bash

git clone https://github.com/RaulHerrera09/churn-customer.behavior-analysis.git
cd YOUR REPO 

2. Create and activate a virtual environment (recommended!):

Bash

Windows:

python -m venv venv

venv\Scripts\activate

macOS / Linux:

python3 -m venv venv

source venv/bin/activate

3. Install the dependencies:

Bash

pip install -r requirements.txt

4. Run the Streamlit application:

Bash

streamlit run app/app.py

And that's it! The application will open automatically in your browser.