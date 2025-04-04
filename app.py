import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from langchain_groq import ChatGroq
import os

# ----------------------------
# 1. Database Setup
# ----------------------------

def init_db():
    conn = sqlite3.connect('employee_data.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS employees (
        Employee_ID TEXT PRIMARY KEY,
        Vibe_Score INTEGER,
        Work_Hours REAL,
        Days_Since_Leave INTEGER,
        Last_Conversation TEXT,
        Status TEXT
    )''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversations (
        Conversation_ID INTEGER PRIMARY KEY AUTOINCREMENT,
        Employee_ID TEXT,
        Message TEXT,
        Response TEXT,
        Timestamp TEXT
    )''')
    
    conn.commit()
    conn.close()

# ----------------------------
# 2. Data Processing
# ----------------------------

def process_uploaded_file(file, date_columns=[]):
    """Handle CSV uploads with date parsing"""
    try:
        df = pd.read_csv(file)
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def calculate_metrics(leave_df):
    """Safe date difference calculation"""
    today = datetime.now().date()
    leave_df = leave_df.copy()
    
    if 'Leave_End_Date' in leave_df.columns:
        leave_df['Days_Since_Leave'] = leave_df['Leave_End_Date'].apply(
            lambda x: (today - x.date()).days 
            if pd.notnull(x) else 999
        )
    return leave_df

# ----------------------------
# 3. Employee Selection Logic
# ----------------------------

def select_employees(vibemeter, leave, activity):
    """Enhanced selection with data validation"""
    try:
        # Merge datasets
        merged = (
            vibemeter.merge(leave, on='Employee_ID', how='left')
            .merge(activity, on='Employee_ID', how='left')
        )
        
        # Apply selection criteria
        selected = merged[
            (merged['Vibe_Score'] <= 2) |
            (merged['Work_Hours'] > 9) |
            (merged['Days_Since_Leave'] > 30)
        ]
        
        return selected[['Employee_ID', 'Vibe_Score', 'Work_Hours', 'Days_Since_Leave']]
    
    except KeyError as e:
        st.error(f"Missing required column: {str(e)}")
        return pd.DataFrame()

# ----------------------------
# 4. Groq Integration
# ----------------------------

def get_groq_response(employee_data, message):
    """Generate context-aware responses"""
    if not st.session_state.get('groq_api_key'):
        return "API key not configured"
    
    try:
        llm = ChatGroq(
            temperature=0.7,
            model_name="mixtral-8x7b-32768",
            api_key=st.session_state.groq_api_key
        )
        
        system_prompt = f"""
        Employee Context:
        - Vibe Score: {employee_data.get('Vibe_Score', 'N/A')}
        - Avg Hours: {employee_data.get('Work_Hours', 'N/A')}
        - Days Since Leave: {employee_data.get('Days_Since_Leave', 'N/A')}
        """
        
        return llm.chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]).content
        
    except Exception as e:
        return f"Error: {str(e)}"

# ----------------------------
# 5. Streamlit Application
# ----------------------------

st.set_page_config(
    page_title="Deloitte Well-being Bot",
    page_icon="🤖",
    layout="wide"
)

init_db()
if 'selected_employees' not in st.session_state:
    st.session_state.selected_employees = pd.DataFrame()

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
# Page: Data Upload
# ----------------------------
if page == "Data Upload":
    st.title("Data Management")
    
    with st.expander("Upload Vibemeter Data"):
        vibe_file = st.file_uploader("Vibemeter CSV", type=['csv'], key='vibe')
        if vibe_file:
            st.session_state.vibemeter = process_uploaded_file(
                vibe_file, ['Response_Date']
            )
    
    with st.expander("Upload Leave Data"):
        leave_file = st.file_uploader("Leave CSV", type=['csv'], key='leave')
        if leave_file:
            st.session_state.leave = calculate_metrics(
                process_uploaded_file(leave_file, ['Leave_Start_Date', 'Leave_End_Date'])
            )
    
    with st.expander("Upload Activity Data"):
        activity_file = st.file_uploader("Activity CSV", type=['csv'], key='activity')
        if activity_file:
            st.session_state.activity = process_uploaded_file(activity_file, ['Date'])

# ----------------------------
# Page: Employee Selection  
# ----------------------------
elif page == "Employee Selection":
    st.title("Employee Selection Engine")
    
    if st.button("Run Selection Algorithm"):
        required_data = ['vibemeter', 'leave', 'activity']
        if all([d in st.session_state for d in required_data]):
            st.session_state.selected_employees = select_employees(
                st.session_state.vibemeter,
                st.session_state.leave,
                st.session_state.activity
            )
            st.success(f"Selected {len(st.session_state.selected_employees)} employees")
        else:
            st.error("Missing required datasets")
    
    if not st.session_state.selected_employees.empty:
        st.dataframe(st.session_state.selected_employees, use_container_width=True)

# ----------------------------
# Page: Conversations
# ----------------------------
elif page == "Conversations":
    st.title("Employee Interactions")
    
    if not st.session_state.selected_employees.empty:
        emp = st.selectbox("Select Employee", st.session_state.selected_employees['Employee_ID'])
        emp_data = st.session_state.selected_employees[
            st.session_state.selected_employees['Employee_ID'] == emp
        ].iloc[0].to_dict()
        
        # Conversation History
        conn = sqlite3.connect('employee_data.db')
        history = pd.read_sql(
            f"SELECT * FROM conversations WHERE Employee_ID = '{emp}'",
            conn
        )
        
        # Display History
        for _, row in history.iterrows():
            st.chat_message("user").write(row['Message'])
            st.chat_message("assistant").write(row['Response'])
        
        # New Message
        if prompt := st.chat_input("Type your message"):
            response = get_groq_response(emp_data, prompt)
            
            # Store conversation
            conn.execute('''
                INSERT INTO conversations 
                (Employee_ID, Message, Response, Timestamp)
                VALUES (?,?,?,?)
            ''', (emp, prompt, response, datetime.now().isoformat()))
            
            # Update employee status
            conn.execute('''
                UPDATE employees SET
                Last_Conversation = ?,
                Status = ?
                WHERE Employee_ID = ?
            ''', (datetime.now().isoformat(), "Engaged", emp))
            
            conn.commit()
            conn.close()
            st.rerun()

# ----------------------------
# Page: Analytics
# ----------------------------
elif page == "Analytics":
    st.title("Employee Well-being Analytics")
    
    conn = sqlite3.connect('employee_data.db')
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Vibe Score Distribution")
        vibe_dist = pd.read_sql("SELECT Vibe_Score, COUNT(*) FROM employees GROUP BY Vibe_Score", conn)
        st.bar_chart(vibe_dist.set_index('Vibe_Score'))
    
    with col2:
        st.subheader("Work Hours Analysis")
        hours_data = pd.read_sql("SELECT Work_Hours FROM employees", conn)
        st.line_chart(hours_data)
    
    st.subheader("Recent Conversations")
    convos = pd.read_sql("SELECT * FROM conversations ORDER BY Timestamp DESC LIMIT 10", conn)
    st.dataframe(convos)

# ----------------------------
# Page: Settings
# ----------------------------
elif page == "Settings":
    st.title("System Configuration")
    
    st.subheader("Groq API Settings")
    api_key = st.text_input("Enter Groq API Key", type="password")
    if api_key:
        st.session_state.groq_api_key = api_key
        st.success("API key configured")
    
    st.subheader("Model Settings")
    model_name = st.selectbox("LLM Model", ["mixtral-8x7b-32768", "llama2-70b-4096"])
    temperature = st.slider("Creativity", 0.0, 1.0, 0.7)

# ----------------------------
# Initialization
# ----------------------------
if __name__ == "__main__":
    st.write("Deloitte Employee Well-being Bot v2.0")
