import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime

now = datetime.now()

def filter_predecessors(df, id_col, predecessor_col):
    predecessors = df[df[id_col] == id][predecessor_col].tolist()[0]
    if predecessors is None:
        return None
    else:
        predecessors_list = [int(x) for x in predecessors.split(',') if x]
        return df[df[id_col].isin(predecessors_list)]

# Ask the user to upload a file
file = st.file_uploader('Upload a file', type=['csv', 'xlsx'])

# If a file is uploaded, read it into a DataFrame
if file is not None:
    df = pd.read_csv(file) if file.type == 'text/csv' else pd.read_excel(file)

    # Display a list of columns in the file
    st.write('Columns in the file:')
    st.write(df.columns)

    # Ask the user to select the ID column
    id_col = st.selectbox('Select the column containing the IDs', df.columns)

    # Ask the user to select the predecessor column
    predecessor_col = st.multiselect('Select the column containing the predecessor IDs', df.columns)

    # Ask the user to enter an ID to check the critical path
    id = st.number_input('Enter an ID to check the critical path')

    # Filter the DataFrame by the predecessors of the specified ID
    filtered_df = filter_predecessors(df, id_col, predecessor_col)

    # If the filtered DataFrame is not empty, display it and a Gantt chart
    if filtered_df is not None:
        filtered_df['Zero_Duration'] = filtered_df['Start'] == filtered_df['Finish']
        filtered_df = filtered_df.sort_values(by=["Finish","Start","Duration_days"], ascending=False)
        zero_duration_tasks = filtered_df[filtered_df["Start"] == filtered_df["Finish"]]

        # Create a Gantt chart of the filtered dataframe
        fig = px.timeline(filtered_df, x_start='Start', x_end='Finish', y=filtered_df[id_col].astype(str), color='Status', hover_name=id_col)

        # Add a title to the chart
        fig.update_layout(title='Project Timeline', font=dict(family='Arial', size=16), width=1200, height=800)

        # Customize the colors
        fig.update_traces(marker=dict(color='rgba(50, 50, 50, 0.8)'), selector=dict(mode='markers'))
        fig.update_traces(marker=dict(color='rgba(100, 100, 100, 0.8)'), selector=dict(mode='lines'))

        fig.add_trace(go.Scatter(x=zero_duration_tasks['Start'], y=zero_duration_tasks[id_col], mode='markers', marker=dict(symbol='star', color='green',size=15), name='Milestones'))

        # Add a vertical line for the current time
        fig.update_layout(xaxis=dict(showgrid=False), shapes=[dict(type='line', xref='x', x0=now, x1=now, yref='paper', y0=0, y1=1, line=dict(color='red', width=1, dash='dot'))])
        # Customize the appearance of the chart
        # Set the x-axis to display ticks every month
        #fig.update_layout(xaxis=dict(tickmode='linear'))

        # Show the chart
        st.plotly_chart(fig)
