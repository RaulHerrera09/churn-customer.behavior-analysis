import streamlit as st
import pandas as pd
import plotly.express as px

# Page configuration
st.set_page_config(page_title="General Overview", layout="wide")

st.title("Customer Overview")
st.markdown("""
    This is the main page of our banking analytics dashboard. 
    It displays a demographic and behavioral profile of the customer base.
    Data sourced from `Bank_marketing.csv`.
""")

# --- Data Loading and Preparation Functions ---


@st.cache_data
def load_overview_data(path):
    try:
        df = pd.read_csv(path)

        # We remove the Naive Bayes model columns which are not relevant for this overview
        df_cleaned = df.filter(regex=r'^(?!Naive_Bayes_).*')

        # Map the attrition flag for clarity
        df_cleaned['Attrition_Label'] = df_cleaned['Attrition_Flag'].map({
            'Existing Customer': 'Active Customer',
            'Attrited Customer': 'Churned Customer'
        })
        # Create a numeric variable for KPIs
        df_cleaned['Attrition_Num'] = df_cleaned['Attrition_Flag'].map({
            'Existing Customer': 0,
            'Attrited Customer': 1
        })
        return df_cleaned

    except FileNotFoundError:
        st.error(f"Error: File not found at {path}.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


# --- Data Loading ---
df_overview = load_overview_data("data/Bank_marketing.csv")

if df_overview is not None:
    # --- Main KPIs ---
    st.header("Global Customer Metrics")

    total_customers = df_overview.shape[0]
    attrition_rate = df_overview['Attrition_Num'].mean() * 100
    avg_trans_amt = df_overview['Total_Trans_Amt'].mean()
    avg_age = df_overview['Customer_Age'].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Attrition (Churn) Rate", f"{attrition_rate:.2f}%")
    col3.metric("Avg. Transaction Amount", f"${avg_trans_amt:,.2f}")
    col4.metric("Average Age", f"{avg_age:.1f} years")

    st.divider()

    # --- Demographic and Behavioral Analysis ---
    st.header("Customer Profile")

    col1, col2 = st.columns(2)

    with col1:
        # Chart 1: Age Distribution
        st.subheader("Age Distribution and Attrition")
        fig_age = px.histogram(df_overview,
                               x="Customer_Age",
                               color="Attrition_Label",
                               marginal="box",
                               barmode="overlay",
                               title="Age Distribution by Customer Status",
                               labels={'Customer_Age': 'Customer Age', 'Attrition_Label': 'Status'})
        st.plotly_chart(fig_age, use_container_width=True)
        st.markdown("""
        **Analysis:** Most of our customers are concentrated between 40 and 50 years old. 
        A slight increase in the proportion of churn is observed in middle-aged customers.
        """)

    with col2:
        # Chart 2: Income Distribution
        st.subheader("Income Distribution")
        # Order income categories (requires a sorting function or list)
        income_order = [
            'Unknown',
            'Less than $40K',
            '$40K - $60K',
            '$60K - $80K',
            '$80K - $120K',
            '$120K+'
        ]

        # Count values and reindex to sort
        income_counts = df_overview['Income_Category'].value_counts().reindex(
            income_order)

        fig_income = px.bar(income_counts,
                            x=income_counts.index,
                            y=income_counts.values,
                            title="Distribution by Income Category",
                            labels={'y': 'Number of Customers', 'x': 'Income Category'})
        st.plotly_chart(fig_income, use_container_width=True)
        st.markdown("""
        **Analysis:** The most common income category is 'Less than $40K', 
        closely followed by middle-income categories. 
        The 'Unknown' category is also significant.
        """)

    col3, col4 = st.columns(2)

    with col3:
        # Chart 3: Card Popularity
        st.subheader("Popularity of Card Types")

        fig_card = px.pie(df_overview,
                          names='Card_Category',
                          title='Customer Distribution by Card Type',
                          hole=0.4)
        fig_card.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_card, use_container_width=True)
        st.markdown("""
        **Analysis:** The 'Blue' card is by far the most popular and represents 
        the vast majority of our customer portfolio.
        """)

    with col4:
        # Chart 4: Transaction Relationship
        st.subheader("Transaction Behavior")
        fig_scatter = px.scatter(df_overview,
                                 x="Total_Trans_Ct",
                                 y="Total_Trans_Amt",
                                 color="Card_Category",
                                 title="Total Amount vs. Transaction Count",
                                 labels={'Total_Trans_Ct': 'Transaction Count',
                                         'Total_Trans_Amt': 'Total Transaction Amount'},
                                 hover_data=['Customer_Age', 'Income_Category'])
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown("""
        **Analysis:** We see a clear linear correlation: more transactions lead to a higher total amount. 
        Customers with 'Silver', 'Gold', and 'Platinum' cards (though few) 
        tend to make higher-value transactions.
        """)

    # --- Show Raw Data ---
    with st.expander("View Raw Marketing Overview Data"):
        st.dataframe(df_overview.drop(columns=['Attrition_Num']))

else:
    st.warning("Could not load overview data for analysis.")
