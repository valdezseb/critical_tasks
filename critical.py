import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import math
import re

def duration_to_days(duration):
    if duration is None:
        return None
    elif duration.endswith('emo'):
        return pd.Timedelta(days=int(math.ceil(float(duration[:-3]) * 20)))
    elif duration.endswith('ew'):
        return pd.Timedelta(days=int(math.ceil(float(duration[:-2]) * 5)))
    elif duration.endswith('m'):
        return pd.Timedelta(days=int(math.ceil(float(duration[:-2]) * 20)))
    elif duration.endswith('w'):
        return pd.Timedelta(days=int(math.ceil(float(duration[:-2]) * 5)))
    else:
        # Check if duration is a valid numeric value
        try:
            return pd.Timedelta(days=int(math.ceil(float(duration[:-2]))))
        except ValueError:
            # If not a valid numeric value, return None
            return None

def days_to_start(df, id_col, start_col, pred_col, status_col):
    """
    Calculates the number of days to start based on the last finish date of predecessors.

    Args:
        df (pandas.DataFrame): The input dataframe.
        id_col (str): The name of the column containing task IDs.
        start_col (str): The name of the column containing task start dates.
        pred_col (str): The name of the column containing predecessor IDs.
        status_col (str): The name of the column containing task status.

    Returns:
        pandas.Series: A series containing the number of days to start for each task.
    """
    # Create a dictionary to store the last finish date of each task
    last_finish_dates = {}

    # Iterate over each row in the dataframe
    for index, row in df.iterrows():
        # Get the task ID and predecessor IDs for the current row
        task_id = row[id_col]
        pred_ids = row[pred_col]

        # If there are no predecessors, set the number of days to start to 0
        if pd.isna(pred_ids):
            df.at[index, 'Days_to_Start'] = 0
        else:
            # Find the latest finish date of all predecessors that have status other than "Complete"
            pred_finish_dates = []
            for pred_id in pred_ids.split(','):
                if pred_id:
                    pred_df = df[(df[id_col] == int(pred_id)) & (df[status_col] != "Complete")].sort_values(by='Finish', ascending=False)
                    if not pred_df.empty:
                        pred_finish_date = pred_df['Finish'].values[0]
                        pred_finish_dates.append(pred_finish_date)

            if pred_finish_dates:
                # If there are predecessors with status other than "Complete", use the latest finish date
                latest_finish_date = max(pred_finish_dates)

                # Calculate the number of days between the latest finish date and the task start date
                task_start_date = row[start_col]
                days_to_start = (task_start_date - latest_finish_date).days

                # Set the number of days to start for the current task
                df.at[index, 'Days_to_Start'] = days_to_start
            else:
                # If all predecessors have status = "Complete", set the number of days to start to 0
                df.at[index, 'Days_to_Start'] = 0

        # Store the last finish date of the current task
        last_finish_dates[task_id] = row['Finish']

    # Return a series containing the number of days to start for each task
    return df['Days_to_Start']

import re

def extract_number(s):
    if isinstance(s, str):
        # split the string by commas
        parts = s.split(',')

        # extract the digits from the parts and return them as a string
        numbers = []
        for part in parts:
            # extract the predecessor ID and ignore the rest of the string
            match = re.search(r'(\d+)(?:FS[+-]\d+)?|\\\d+$|<.+?IMS\\\d+|<.+?Components\\\d+|<.+?Systems\\\d+|\d+', str(part))
            if match:
                if '\\' in match.group(0):
                    # extract the last sequence of digits after the backslash
                    digits = re.findall(r'\d+', match.group(0))[-1]
                elif match.group(0).startswith('<'):
                    # extract the last sequence of digits after the backslash
                    digits = re.findall(r'\d+', match.group(0))[-1]
                else:
                    digits = match.group(1)
                numbers.extend(digits.split(','))

        if numbers:
            # remove duplicates and return as a string
            return ', '.join(set(numbers))
        else:
            return None
    elif isinstance(s, int):
        # return the integer value as a string
        return str(s)
    else:
        return None
        # convert non-string and non-integer



def filter_predecessors(df, id):
    predecessors = df[df['ID'] == id]['new_Predecessors'].tolist()[0]
    if predecessors is None:
        return None
    else:
        predecessors_list = [int(x) for x in predecessors.split(',') if x]
        return df[df['ID'].isin(predecessors_list)]

#@st.cache_data
def load_data(uploaded_file):
    df = pd.read_excel(uploaded_file, engine="openpyxl")
    return df


def transform_data(df):
    df['New_Duration'] = df['Dur'].apply(duration_to_days)
    df['new_Predecessors'] = df['Predecessors'].apply(extract_number)
    #df['Days_to_Start'] = days_to_start(df, 'ID', 'Start', 'new_Predecessors', 'Status')
    
    return df

def filter_predecessors(df, id):
    predecessors = df[df['ID'] == id]['new_Predecessors'].tolist()

    if not predecessors or predecessors[0] is None:
        # Handle empty predecessors list or None
        return None
    else:
        predecessors_list = [int(x) for x in predecessors[0].split(',') if x]
        return df[df['ID'].isin(predecessors_list)]


# Streamlit code for file upload, column selection, and ID input
uploaded_file = st.file_uploader("Choose a xlsx file", type="xlsx")
now = datetime.now()

if uploaded_file is not None:
    # Load data
    df = load_data(uploaded_file)
    #df = pd.read_excel(uploaded_file, engine="openpyxl")

    # Column selection
    start_column = st.selectbox('Select Start Date Column', df.columns, index=0 if 'Start' in df.columns else None)
    finish_column = st.selectbox('Select Finish Date Column', df.columns, index=1 if 'Finish' in df.columns else None)
    #duration_column = st.selectbox('Select Duration Column', df.columns, index=2 if 'Duration' in df.columns else None)
    status_column = st.selectbox('Select Status Column (or color label)', df.columns, index=3 if 'Status' in df.columns else None)


    if start_column is not None and finish_column is not None and status_column is not None:
        # Data transformation
        #df['New_Duration'] = df[duration_column].apply(duration_to_days)
        df['new_Predecessors'] = df['Predecessors'].apply(extract_number)
    else:
        st.write("Fill names.")



    task_id = st.text_input('Enter Task ID:')

    if task_id:
        # Check if task ID is a valid integer
        if not task_id.isdigit():
            st.write("Invalid task ID:", task_id)
            task_id = None
        else:
            task_id = int(task_id)


    if task_id:
        filtered_df = filter_predecessors(df, task_id)

        if filtered_df is not None:
            filtered_df = filtered_df.sort_values(by=[finish_column, start_column], ascending=False)

            # Add a title to the chart
            filtered_df[start_column] = pd.to_datetime(filtered_df[start_column])
            filtered_df[finish_column] = pd.to_datetime(filtered_df[finish_column])

            # Create and display the Gantt chart
            fig = px.timeline(filtered_df, x_start=start_column, x_end=finish_column, y=filtered_df["ID"].astype(str),
                               color=status_column, hover_name='ID')
            fig.update_layout(title='Project Timeline', font=dict(family='Arial', size=16), width=1200, height=800)

            # Customize the colors
            fig.update_traces(marker=dict(color='rgba(50, 50, 50, 0.8)'), selector=dict(mode='markers'))
            fig.update_traces(marker=dict(color='rgba(100, 100, 100, 0.8)'), selector=dict(mode='lines'))

            zero_duration_tasks = filtered_df[filtered_df[start_column] == filtered_df[finish_column]]

            status_colors = {
                    'Future Task': 'gray',
                    'In Progress': 'blue',
                    'Complete': 'green'
                }





            # Check if any task is not complete
            fig.add_trace(go.Scatter(x=zero_duration_tasks[start_column], y=zero_duration_tasks['ID'], mode='markers',
                            marker=dict(symbol='star', color=[status_colors[status] for status in zero_duration_tasks[status_column] ], size=15), name='Milestones'))








            # Add a vertical line for the current time
            fig.update_layout(xaxis=dict(showgrid=False),
                               shapes=[dict(type='line', xref='x', x0=now, x1=now, yref='paper', y0=0, y1=1,
                                           line=dict(color='red', width=1, dash='dot'))])

            # Display Gantt chart and filtered DataFrame
            st.markdown('## Project Analysis')

            with st.container():
                st.subheader('Gantt Chart')
                st.plotly_chart(fig)

            with st.container():
                st.subheader('Filtered DataFrame')
                st.dataframe(filtered_df)
        else:
            st.write("No tasks found for the selected ID.")
