import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime
import openpyxl


def get_critical_path(df, id_col, predecessor_col, start_col, end_col, id):
    # Get predecessors for the specified ID
    predecessors = df[df[id_col]==id][predecessor_col].tolist()[0]
    
    # Initialize critical path list
    critical_path = [id]
    
    # Recursively get critical path 
    def get_crit_path(id):
        preds = df[df[id_col]==id][predecessor_col].tolist()[0]
        for p in preds.split(','):
            if p=='': continue
            p = int(p)
            if p not in critical_path:
                critical_path.append(p)
                get_crit_path(p)
        
    get_crit_path(id)
        
    # Return filtered dataframe  
    return df[df[id_col].isin(critical_path)]

# Allow user to upload file  
file = st.file_uploader('Upload file')
if file:
    df = pd.read_excel(file)
    
    # Get input
    id_col = st.selectbox('ID col', df.columns)
    predecessor_col = st.selectbox('Predecessors col', df.columns)
    start_col = st.selectbox('Start col', df.columns)
    end_col = st.selectbox('End col', df.columns)
    id = st.number_input('Enter ID', min_value=0, max_value=df[id_col].max())
    
    # Get and display critical path
    crit_df = get_critical_path(df, id_col, predecessor_col, start_col, end_col, id)
    fig = px.timeline(crit_df, x_start=start_col, x_end=end_col, y=id_col)
    st.plotly_chart(fig)
