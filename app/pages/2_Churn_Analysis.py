import streamlit as st
import pandas as pd
import plotly.express as px

# Page configuration
st.set_page_config(page_title="Churn Analysis", layout="wide")

st.title("Customer Churn Analysis")
st.markdown(
    "This page analyzes the factors that contribute to customer churn, based on `churn_modelling.csv`.")

# --- Data Loading Functions ---
# We use @st.cache_data so data is loaded only once


@st.cache_data
def load_data(path):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        st.error(
            f"Error: File not found at {path}. Make sure the file is in the 'data' directory.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


# --- Data Loading ---
# The path is relative to the project root (where you run `streamlit run app.py`)
df_churn = load_data("data/Churn_modelling.csv")

if df_churn is not None:
    # --- Main KPIs ---
    st.header("Key Performance Indicators (KPIs)")

    # Calculate Churn Rate
    total_customers = df_churn.shape[0]
    churned_customers = df_churn[df_churn['Exited'] == 1].shape[0]
    churn_rate = (churned_customers / total_customers) * 100

    # Calculate Average Age (Churn vs. No Churn)
    avg_age_churned = df_churn[df_churn['Exited'] == 1]['Age'].mean()
    avg_age_stayed = df_churn[df_churn['Exited'] == 0]['Age'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Churn Rate", f"{churn_rate:.2f}%")
    col3.metric("Avg. Age (Churned)", f"{avg_age_churned:.1f} years")

    st.divider()

    # --- Interactive Visualizations ---
    st.header("Visual Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Chart 1: Age Distribution vs. Churn
        st.subheader("Age Distribution by Churn")
        fig_age = px.histogram(df_churn,
                               x="Age",
                               color="Exited",
                               barmode="overlay",
                               marginal="box",
                               title="Age Distribution (0 = Stays, 1 = Exits)")
        st.plotly_chart(fig_age, use_container_width=True)
        st.markdown("""
        **Analysis:** A clear trend is observed where older customers (especially between 45-65 years) 
        have a significantly higher churn rate ('Exited' = 1).
        """)

    with col2:
        # Chart 2: Churn by Country
        st.subheader("Churn Rate by Country")
        # We calculate the % churn by country
        churn_by_geo = df_churn.groupby(
            'Geography')['Exited'].mean().reset_index()
        churn_by_geo['Exited'] = churn_by_geo['Exited'] * \
            100  # Convert to percentage

        fig_geo = px.bar(churn_by_geo,
                         x='Geography',
                         y='Exited',
                         title='Churn Percentage by Country',
                         labels={'Geography': 'Country', 'Exited': '% Churn'})
        st.plotly_chart(fig_geo, use_container_width=True)
        st.markdown("""
        **Analysis:** Germany shows a much higher churn rate than France or Spain. 
        This suggests there may be specific issues in the German market.
        """)

    col3, col4 = st.columns(2)

    with col3:
        # Chart 3: Balance vs. Churn
        st.subheader("Balance Distribution by Churn")
        fig_balance = px.box(df_churn,
                             x="Exited",
                             y="Balance",
                             color="Exited",
                             title="Balance Distribution (0 = Stays, 1 = Exits)")
        st.plotly_chart(fig_balance, use_container_width=True)
        st.markdown("""
        **Analysis:** Customers who leave ('Exited' = 1) tend to have higher account balances. 
        Interestingly, customers with a $0 balance (possibly inactive or pass-through accounts) have a high retention rate.
        """)

    with col4:
        # Chart 4: Churn by Active Member
        st.subheader("Impact of Active Membership")
        churn_by_active = df_churn.groupby('IsActiveMember')[
            'Exited'].mean().reset_index()
        churn_by_active['Exited'] = churn_by_active['Exited'] * 100
        churn_by_active['IsActiveMember'] = churn_by_active['IsActiveMember'].map(
            {1: 'Active', 0: 'Inactive'})

        fig_active = px.pie(churn_by_active,
                            names='IsActiveMember',
                            values='Exited',
                            title='Churn Percentage (Active vs. Inactive Members)',
                            hole=0.4)
        st.plotly_chart(fig_active, use_container_width=True)
        st.markdown("""
        **Analysis:** As expected, inactive members 
        have a much higher probability of leaving the bank than active members.
        """)

    # --- Show Raw Data ---
    with st.expander("View Raw Churn Data"):
        st.dataframe(df_churn.drop(
            columns=['RowNumber', 'CustomerId', 'Surname']))

else:
    st.warning("Could not load churn data for analysis.")
