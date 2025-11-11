import streamlit as st
import pandas as pd
import plotly.express as px

# Page configuration
st.set_page_config(page_title="Loan Analysis", layout="wide")

st.title("Personal Loan Insights")
st.markdown("""
    This section identifies the profile of the ideal customer who accepts a personal loan. 
    The goal is to help the marketing team focus their efforts more efficiently.
""")

# --- Loading and Preparation Functions ---


@st.cache_data
def load_and_prep_data(path):
    try:
        df = pd.read_csv(path)

        # --- Professional Cleaning and Mapping ---
        # We use the data dictionary to make the analysis readable
        df['Education_Label'] = df['Education'].map({
            1: 'Undergrad',
            2: 'Graduate',
            3: 'Advanced/Professional'
        })

        # Map the target variable for clarity in charts
        df['Personal_Loan_Label'] = df['Personal Loan'].map({
            0: 'Did Not Accept',
            1: 'Accepted'
        })

        # Convert binary variables to clear labels
        df['CD_Account_Label'] = df['CD Account'].map({0: 'No', 1: 'Yes'})
        df['Online_Label'] = df['Online'].map({0: 'No', 1: 'Yes'})
        df['CreditCard_Label'] = df['CreditCard'].map({0: 'No', 1: 'Yes'})

        return df

    except FileNotFoundError:
        st.error(f"Error: File not found at {path}.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


# --- Data Loading ---
df_loan = load_and_prep_data("data/Bank_Loan_Modelling - Data.csv")

if df_loan is not None:
    # --- Main KPIs ---
    st.header("Key Campaign Metrics")

    total_customers = df_loan.shape[0]
    accepted_count = df_loan['Personal Loan'].sum()
    conversion_rate = (accepted_count / total_customers) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Customers in Campaign", f"{total_customers:,}")
    col2.metric("Loans Accepted", f"{accepted_count:,}")
    col3.metric("Conversion Rate", f"{conversion_rate:.2f}%")

    st.markdown(
        f"Only **{conversion_rate:.2f}%** of customers accepted the loan. We can improve this.")
    st.divider()

    # --- Profile Analysis ---
    st.header("Profile of the Customer Who Accepts")

    col1, col2 = st.columns(2)

    with col1:
        # Chart 1: Income
        st.subheader("Income vs. Loan Acceptance")
        fig_income = px.box(df_loan,
                            x='Personal_Loan_Label',
                            y='Income',
                            color='Personal_Loan_Label',
                            title="Income Distribution",
                            labels={'Personal_Loan_Label': 'Loan Acceptance', 'Income': 'Annual Income (in $000)'})
        st.plotly_chart(fig_income, use_container_width=True)
        st.markdown("""
        **Observation:** There is a drastic difference. Customers who accept the loan 
        have significantly higher annual incomes.
        """)

    with col2:
        # Chart 2: Education Level
        st.subheader("Conversion Rate by Education Level")
        # Calculate % conversion by group
        education_conversion = df_loan.groupby('Education_Label')[
            'Personal Loan'].mean().reset_index()
        education_conversion['Personal Loan'] = education_conversion['Personal Loan'] * 100

        fig_edu = px.bar(education_conversion.sort_values('Personal Loan', ascending=False),
                         x='Education_Label',
                         y='Personal Loan',
                         title="Conversion Rate by Education",
                         labels={'Education_Label': 'Education Level', 'Personal Loan': 'Conversion Rate (%)'})
        st.plotly_chart(fig_edu, use_container_width=True)
        st.markdown("""
        **Observation:** Customers with 'Advanced/Professional' and 'Graduate' education 
        have a much higher conversion rate than 'Undergrad'.
        """)

    col3, col4 = st.columns(2)

    with col3:
        # Chart 3: CD Account
        st.subheader("Impact of having a Certificate of Deposit (CD)")
        cd_conversion = df_loan.groupby('CD_Account_Label')[
            'Personal Loan'].mean().reset_index()
        cd_conversion['Personal Loan'] = cd_conversion['Personal Loan'] * 100

        fig_cd = px.pie(cd_conversion,
                        names='CD_Account_Label',
                        values='Personal Loan',
                        title="Conversion Rate (Customers with/without CD)",
                        hole=0.4,
                        labels={'CD_Account_Label': 'Has CD Account'})
        st.plotly_chart(fig_cd, use_container_width=True)
        st.markdown("""
        **Key Observation:** Customers who **already have a Certificate of Deposit (CD)** have an extremely high conversion rate. This is a very strong indicator!
        """)

    with col4:
        # Chart 4: Avg Credit Card Spending (CCAvg)
        st.subheader("Average CC Spending vs. Acceptance")
        fig_ccavg = px.box(df_loan,
                           x='Personal_Loan_Label',
                           y='CCAvg',
                           color='Personal_Loan_Label',
                           title="Average CC Spending Distribution",
                           labels={'Personal_Loan_Label': 'Loan Acceptance', 'CCAvg': 'Avg. CC Spending (in $000)'})
        st.plotly_chart(fig_ccavg, use_container_width=True)
        st.markdown("""
        **Observation:** Similar to income, customers who accept the loan 
        tend to have higher average monthly spending on their credit cards.
        """)

    st.divider()

    # --- Business Recommendation ---
    st.header("Marketing Recommendation")
    st.info("""
    **Ideal Customer Profile for the Next Campaign:**
    
    To maximize the Return on Investment (ROI) of the next loan campaign, it is recommended to focus efforts on customers who meet the following profile:
    
    * **High Income:** Customers with annual income over $80K (approx.).
    * **Education Level:** Customers with 'Graduate' or 'Advanced/Professional' degrees.
    * **Engaged Customers:** The strongest indicator. Customers who **already have a Certificate of Deposit (CD)** with the bank.
    * **High CC Spending:** Customers with an average credit card spend (CCAvg) over $2.5K per month.
    
    Customers with low income and 'Undergrad' education level have an almost zero conversion probability and should not be the focus.
    """)

    # --- Show Raw Data ---
    with st.expander("View Raw Loan Analysis Data"):
        st.dataframe(df_loan.drop(columns=['Education', 'Personal Loan']))

else:
    st.warning("Could not load loan data for analysis.")
