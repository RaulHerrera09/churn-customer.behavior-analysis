import streamlit as st

# --- Page Configuration ---
# st.set_page_config MUST be the first Streamlit command in your script.
st.set_page_config(
    page_title="Bank Analytics Dashboard",
    page_icon="🏦",  # Bank icon
    layout="wide",   # Use a wide layout by default
    initial_sidebar_state="expanded",  # Keep the sidebar open
    menu_items={

        'Get Help': 'https://www.linkedin.com/in/raul-herrera-delgadillo-384b6123b/',
        'Report a bug': "https://github.com/RaulHerrera09/churn-customer.behavior-analysis.git",
        'About': """
        ## Bank Simulation Dashboard 
        
        This is an interactive dashboard built with Streamlit 
        to simulate data analysis in a digital bank.
        
        It demonstrates EDA, data visualization, and predictive modeling.
        """
    }
)


st.title("Bank Simulation Dashboard")

st.divider()

st.header("Welcome to this Project")
st.markdown("""
This interactive dashboard is a portfolio project that demonstrates a complete data analysis workflow, 
simulating the role of a Data Analyst in a modern digital bank.

Here you will find exploratory analysis, interactive visualizations, and a Machine Learning model 
to predict customer behavior.
""")


st.info("Use the **menu in the sidebar** to navigate to the different analysis sections.")

st.subheader("Available Sections:")
st.markdown("""
* **1. Overview:** Demographic profile of the customer base.
* **2. Churn Analysis:** KPIs and visualizations on why customers leave.
* **3. Loan Insights:** Profile of the customer who accepts personal loans.
* **4. Predictive Model:** A Machine Learning model that predicts a customer's 
    probability of churning, along with an interactive simulator.
""")

st.divider()
