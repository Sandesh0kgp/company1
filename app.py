import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from langchain_groq import ChatGroq
import os

# Page configuration
st.set_page_config(
    page_title="Deloitte Employee Well-being Bot",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'selected_employees' not in st.session_state:
    st.session_state.selected_employees = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {}
if 'datasets_loaded' not in st.session_state:
    st.session_state.datasets_loaded = False
if 'reports' not in st.session_state:
    st.session_state.reports = {}
if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = ""

# Data loading functions
def load_sample_data():
    """Generate sample datasets for demonstration"""
    # Sample Vibemeter dataset
    vibemeter_data = pd.DataFrame({
        'Employee_ID': [f'EMP{i:04d}' for i in range(1, 501)],
        'Response_Date': [datetime.now().date() - timedelta(days=i) for i in range(500)],
        'Vibe_Score': np.random.randint(1, 6, 500)
    })
    
    # Sample Leave dataset
    leave_data = pd.DataFrame({
        'Employee_ID': [f'EMP{i:04d}' for i in range(1, 501)],
        'Leave_Start_Date': [datetime.now().date() - timedelta(days=30 + i) for i in range(500)],
        'Leave_End_Date': [datetime.now().date() - timedelta(days=25 + i) for i in range(500)]
    })
    
    # Sample Activity Tracker dataset
    activity_data = pd.DataFrame({
        'Employee_ID': [f'EMP{i:04d}' for i in range(1, 501)],
        'Date': [datetime.now().date() - timedelta(days=i) for i in range(500)],
        'Work_Hours': np.random.randint(6, 12, 500)
    })
    
    return vibemeter_data, leave_data, activity_data

# Employee selection logic
def select_employees(vibemeter_df, leave_df, activity_df):
    """Select employees based on well-being criteria"""
    today = datetime.now().date()
    
    # Calculate metrics
    leave_df['Days_Since_Leave'] = (today - leave_df['Leave_End_Date']).dt.days
    activity_agg = activity_df.groupby('Employee_ID')['Work_Hours'].mean().reset_index()
    
    # Merge datasets
    merged_df = vibemeter_df.merge(leave_df, on='Employee_ID').merge(activity_agg, on='Employee_ID')
    
    # Apply selection criteria
    selected = merged_df[
        (merged_df['Vibe_Score'] <= 2) | 
        (merged_df['Work_Hours'] > 9) |
        (merged_df['Days_Since_Leave'] > 30)
    ]
    
    return selected[['Employee_ID', 'Vibe_Score', 'Work_Hours', 'Days_Since_Leave']]

# Groq LLM integration
def generate_response(employee_id, message, employee_data):
    """Generate empathetic response using Groq"""
    if not st.session_state.groq_api_key:
        return "Please configure Groq API key in Settings"
    
    try:
        groq_llm = ChatGroq(
            temperature=0.7,
            model_name="mixtral-8x7b-32768",
            api_key=st.session_state.groq_api_key
        )
        
        system_prompt = f"""You are an empathetic HR assistant. Employee context:
        - Vibe Score: {employee_data['Vibe_Score']}
        - Avg Work Hours: {employee_data['Work_Hours']}
        - Days Since Last Leave: {employee_data['Days_Since_Leave']}"""
        
        response = groq_llm.chat(messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ])
        
        return response.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Home", 
    "Employee Selection", 
    "Data Analysis", 
    "Employee Interaction",
    "Reports & Insights",
    "Settings"
])

# Main content
if page == "Home":
    st.title("Deloitte Employee Well-being Bot")
    st.write("Automated employee well-being tracking and engagement system")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("""
        ### Key Features:
        - Automated employee selection
        - AI-powered conversations
        - Real-time analytics
        - HR escalation system
        - Daily reporting
        """)
        
    with col2:
        if st.button("Load Sample Data"):
            vibemeter, leave, activity = load_sample_data()
            st.session_state.vibemeter = vibemeter
            st.session_state.leave = leave
            st.session_state.activity = activity
            st.session_state.datasets_loaded = True
            st.success("Sample data loaded!")
            
            # Run initial selection
            st.session_state.selected_employees = select_employees(
                st.session_state.vibemeter,
                st.session_state.leave,
                st.session_state.activity
            )

elif page == "Employee Selection":
    st.title("Employee Selection")
    
    if not st.session_state.datasets_loaded:
        st.warning("Load data first from Home page")
    else:
        st.write("### Selected Employees for Engagement")
        st.dataframe(st.session_state.selected_employees)
        
        st.write("### Selection Criteria")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Employees", 500)
            st.metric("Negative Vibe Threshold", "Score ≤ 2")
        with col2:
            st.metric("Work Hour Threshold", ">9 hours/day")
            st.metric("Leave Threshold", ">30 days since last")

elif page == "Data Analysis":
    st.title("Data Analysis")
    
    if not st.session_state.datasets_loaded:
        st.warning("Load data first from Home page")
    else:
        st.write("### Vibe Distribution")
        fig, ax = plt.subplots()
        sns.histplot(st.session_state.vibemeter['Vibe_Score'], bins=5, kde=True)
        st.pyplot(fig)
        
        st.write("### Workload vs Vibe Correlation")
        corr_data = st.session_state.vibemeter.merge(
            st.session_state.activity.groupby('Employee_ID')['Work_Hours'].mean().reset_index(),
            on='Employee_ID'
        )
        fig, ax = plt.subplots()
        sns.scatterplot(x='Work_Hours', y='Vibe_Score', data=corr_data)
        st.pyplot(fig)

elif page == "Employee Interaction":
    st.title("Employee Interaction")
    
    if not st.session_state.selected_employees.empty:
        employee = st.selectbox("Select Employee", st.session_state.selected_employees['Employee_ID'])
        
        # Initialize chat history
        if employee not in st.session_state.chat_history:
            st.session_state.chat_history[employee] = []
            
        # Display chat history
        st.write("### Conversation History")
        for msg in st.session_state.chat_history[employee]:
            st.markdown(f"**{msg['role']}**: {msg['content']}")
            
        # Input and send message
        message = st.text_input("Enter your message")
        if st.button("Send"):
            employee_data = st.session_state.selected_employees[
                st.session_state.selected_employees['Employee_ID'] == employee
            ].iloc[0].to_dict()
            
            response = generate_response(employee, message, employee_data)
            
            st.session_state.chat_history[employee].append({
                "role": "User",
                "content": message
            })
            st.session_state.chat_history[employee].append({
                "role": "Bot",
                "content": response
            })
            st.experimental_rerun()
    else:
        st.warning("No employees selected. Run selection first.")

elif page == "Reports & Insights":
    st.title("Daily Reports")
    
    if not st.session_state.datasets_loaded:
        st.warning("Load data first from Home page")
    else:
        st.write("### Key Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Employees Selected", len(st.session_state.selected_employees))
        col2.metric("Avg Vibe Score", round(st.session_state.selected_employees['Vibe_Score'].mean(), 1))
        col3.metric("Avg Work Hours", round(st.session_state.selected_employees['Work_Hours'].mean(), 1))
        
        st.write("### Recommended Actions")
        st.write("1. Schedule team-building activities for Department A")
        st.write("2. Review workload distribution in Department B")
        st.write("3. Conduct recognition workshops for Department C")

elif page == "Settings":
    st.title("System Settings")
    
    st.write("### Groq API Configuration")
    api_key = st.text_input("Enter Groq API Key", type="password")
    if api_key:
        st.session_state.groq_api_key = api_key
        st.success("API key configured!")
    
    st.write("### Model Settings")
    model_name = st.selectbox("Select LLM Model", ["mixtral-8x7b-32768", "llama3-70b-8192"])
    temperature = st.slider("Response Creativity", 0.0, 1.0, 0.7)
    
if __name__ == "__main__":
    st.write("Deloitte Employee Well-being Bot - v1.0")
