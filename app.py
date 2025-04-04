import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from langchain_groq import ChatGroq
from transformers import AutoModelForSequenceClassification, pipeline
import torch
import os

# ----------------------------
# 1. Database Setup
# ----------------------------

def init_db():
    conn = sqlite3.connect('employee_data.db')
    cursor = conn.cursor()
    
    # Create tables for all 6 datasets
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS employees (
        Employee_ID TEXT PRIMARY KEY,
        Vibe_Score INTEGER,
        Work_Hours REAL,
        Days_Since_Leave INTEGER,
        Performance_Rating REAL,
        Last_Award_Date TEXT,
        Onboarding_Score INTEGER,
        Last_Updated TEXT
    )''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversations (
        Conversation_ID INTEGER PRIMARY KEY AUTOINCREMENT,
        Employee_ID TEXT,
        Message TEXT,
        Response TEXT,
        Sentiment TEXT,
        Timestamp TEXT
    )''')
    
    conn.commit()
    conn.close()

# ----------------------------
# 2. Data Processing (All 6 datasets)
# ----------------------------

def safe_date_conversion(df, date_cols):
    """Handle date conversions with NaN/Null checks"""
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df[col] = df[col].apply(lambda x: x if pd.notnull(x) else None)
    return df

def load_and_process(file, date_cols=[]):
    """Process uploaded CSV files"""
    try:
        df = pd.read_csv(file)
        return safe_date_conversion(df, date_cols)
    except Exception as e:
        st.error(f"Error processing {file.name}: {str(e)}")
        return None

# ----------------------------
# 3. Models Setup
# ----------------------------

# Fine-tuned model for sentiment analysis
def load_finetuned_model():
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
    return pipeline("text-classification", model=model)

# Groq LLM for response generation
def init_groq():
    return ChatGroq(
        temperature=0.7,
        model_name="mixtral-8x7b-32768",
        api_key=st.session_state.get('groq_api_key', '')
    )

# ----------------------------
# 4. Core Functions
# ----------------------------

def calculate_metrics(dataframes):
    """Process all 6 datasets and calculate key metrics"""
    dfs = {}
    
    # Process each dataset
    dfs['vibemeter'] = dataframes['vibemeter'][['Employee_ID', 'Vibe_Score', 'Response_Date']]
    
    # Leave data
    dfs['leave'] = dataframes['leave']
    dfs['leave']['Days_Since_Leave'] = (datetime.now().date() - dfs['leave']['Leave_End_Date']).dt.days
    dfs['leave'].fillna({'Days_Since_Leave': 999}, inplace=True)
    
    # Activity data
    dfs['activity'] = dataframes['activity'].groupby('Employee_ID')['Work_Hours'].mean().reset_index()
    
    # Performance data
    dfs['performance'] = dataframes['performance'][['Employee_ID', 'Performance_Rating']]
    
    # Rewards data
    dfs['rewards'] = dataframes['rewards']
    dfs['rewards']['Last_Award_Date'] = pd.to_datetime(dfs['rewards']['Award_Date']).dt.date
    
    # Onboarding data
    dfs['onboarding'] = dataframes['onboarding'][['Employee_ID', 'Onboarding_Experience']]
    
    # Merge all datasets
    merged = dfs['vibemeter'].merge(dfs['leave'], on='Employee_ID') \
                             .merge(dfs['activity'], on='Employee_ID') \
                             .merge(dfs['performance'], on='Employee_ID') \
                             .merge(dfs['rewards'], on='Employee_ID') \
                             .merge(dfs['onboarding'], on='Employee_ID')
    
    return merged

def select_employees(merged_df):
    """Enhanced selection logic using all metrics"""
    selected = merged_df[
        (merged_df['Vibe_Score'] <= 2) |
        (merged_df['Work_Hours'] > 9) |
        (merged_df['Days_Since_Leave'] > 30) |
        (merged_df['Performance_Rating'] < 3) |
        (merged_df['Onboarding_Experience'] < 3)
    ]
    return selected

# ----------------------------
# 5. Streamlit UI
# ----------------------------

st.set_page_config(
    page_title="Deloitte Well-being Bot",
    page_icon="🤖",
    layout="wide"
)
init_db()

# Sidebar Navigation
pages = {
    "Data Upload": "📥",
    "Employee Selection": "👥",
    "Conversations": "💬",
    "Analytics": "📊",
    "Settings": "⚙️"
}
page = st.sidebar.radio("Navigation", list(pages.keys()), format_func=lambda x: f"{pages[x]} {x}")

# ----------------------------
# Page: Data Upload (All 6 datasets)
# ----------------------------
if page == "Data Upload":
    st.title("Data Management")
    
    datasets = {
        "vibemeter": ("Vibemeter Data", ['Response_Date']),
        "leave": ("Leave Data", ['Leave_Start_Date', 'Leave_End_Date']),
        "activity": ("Activity Data", ['Date']),
        "performance": ("Performance Data", []),
        "rewards": ("Rewards Data", ['Award_Date']),
        "onboarding": ("Onboarding Data", ['Joining_Date'])
    }
    
    for key, (name, date_cols) in datasets.items():
        with st.expander(f"Upload {name}"):
            file = st.file_uploader(name, type=['csv'], key=key)
            if file:
                st.session_state[key] = load_and_process(file, date_cols)
                st.success(f"{name} loaded successfully!")

# ----------------------------
# Page: Employee Selection
# ----------------------------
elif page == "Employee Selection":
    st.title("Employee Selection Engine")
    
    if st.button("Run Selection Algorithm"):
        required = ['vibemeter', 'leave', 'activity', 'performance', 'rewards', 'onboarding']
        if all(key in st.session_state for key in required):
            merged = calculate_metrics(st.session_state)
            st.session_state.selected = select_employees(merged)
            st.success(f"Selected {len(st.session_state.selected)} employees")
        else:
            st.error("Missing required datasets!")
    
    if 'selected' in st.session_state:
        st.dataframe(st.session_state.selected, use_container_width=True)

# ----------------------------
# Page: Conversations
# ----------------------------
elif page == "Conversations":
    st.title("Employee Interactions")
    
    if 'selected' in st.session_state and not st.session_state.selected.empty:
        emp = st.selectbox("Select Employee", st.session_state.selected['Employee_ID'])
        emp_data = st.session_state.selected[st.session_state.selected['Employee_ID'] == emp].iloc[0]
        
        # Load models
        sentiment_model = load_finetuned_model()
        groq_llm = init_groq()
        
        # Chat interface
        if prompt := st.chat_input("Type your message"):
            # Generate response
            response = groq_llm.chat([{"role": "user", "content": prompt}]).content
            
            # Analyze sentiment
            sentiment = sentiment_model(prompt)[0]['label']
            
            # Store conversation
            conn = sqlite3.connect('employee_data.db')
            conn.execute('''
                INSERT INTO conversations 
                (Employee_ID, Message, Response, Sentiment, Timestamp)
                VALUES (?,?,?,?,?)
            ''', (emp, prompt, response, sentiment, datetime.now().isoformat()))
            conn.commit()
            conn.close()
            
            st.rerun()
        
        # Display history
        conn = sqlite3.connect('employee_data.db')
        history = pd.read_sql(f"SELECT * FROM conversations WHERE Employee_ID = '{emp}'", conn)
        for _, row in history.iterrows():
            st.chat_message("user").write(row['Message'])
            st.chat_message("assistant").write(row['Response'])

# ----------------------------
# Page: Analytics
# ----------------------------
elif page == "Analytics":
    st.title("Comprehensive Analytics")
    
    conn = sqlite3.connect('employee_data.db')
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Well-being Distribution")
        df = pd.read_sql("SELECT Vibe_Score, COUNT(*) FROM employees GROUP BY Vibe_Score", conn)
        st.bar_chart(df.set_index('Vibe_Score'))
    
    with col2:
        st.subheader("Performance Analysis")
        df = pd.read_sql("SELECT Performance_Rating, AVG(Vibe_Score) FROM employees GROUP BY Performance_Rating", conn)
        st.line_chart(df.set_index('Performance_Rating'))
    
    st.subheader("Recent Escalations")
    df = pd.read_sql("SELECT * FROM conversations WHERE Sentiment = 'NEGATIVE'", conn)
    st.dataframe(df)

# ----------------------------
# Page: Settings
# ----------------------------
elif page == "Settings":
    st.title("System Configuration")
    
    st.subheader("API Keys")
    st.session_state.groq_api_key = st.text_input("Groq API Key", type="password")
    
    st.subheader("Model Settings")
    st.selectbox("Fine-tuned Model", ["distilbert-base-uncased", "bert-base-uncased"])
    st.slider("Response Temperature", 0.0, 1.0, 0.7)

# ----------------------------
# Initialization
# ----------------------------
if __name__ == "__main__":
    st.write("Deloitte Employee Well-being Analytics System v3.0")
