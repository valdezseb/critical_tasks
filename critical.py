

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import math
import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.markers import MarkerStyle

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
    elif duration.endswith('ed'):
        return pd.Timedelta(days=int(math.ceil(float(duration[:-2]) * 1)))
    else:
        return pd.Timedelta(days=int(math.ceil(float(duration[:-2]))))

@st.cache_data
def days_to_start(df, id_col, start_col, pred_col, status_col, finish_column):
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
                if pred_id.strip():  # skip empty or space characters
                    pred_df = df[(df[id_col] == int(pred_id)) & (df[status_col] != "Complete")].sort_values(by=finish_column, ascending=False)
                    if not pred_df.empty:
                        pred_finish_date = pred_df[finish_column].values[0]
                        pred_finish_dates.append(pred_finish_date)

            if pred_finish_dates:
                # If there are predecessors with status other than "Complete", use the latest finish date
                latest_finish_date = max(pred_finish_dates)
                latest_finish_date = pd.to_datetime(latest_finish_date, errors='coerce')

                # Calculate the number of days between the latest finish date and the task start date
                task_start_date = row[start_col]
                task_start_date = pd.to_datetime(row[start_col], errors='coerce')
                days_to_start = (task_start_date - latest_finish_date).days

                # Set the number of days to start for the current task
                df.at[index, 'Days_to_Start'] = days_to_start
            else:
                # If all predecessors have status = "Complete", set the number of days to start to 0
                df.at[index, 'Days_to_Start'] = 0

        # Store the last finish date of the current task
        last_finish_dates[task_id] = row[finish_column]
        

    # Return a series containing the number of days to start for each task
    return df['Days_to_Start']


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


#@st.cache_data
def load_data(uploaded_file):
    #df = pd.read_excel(uploaded_file, engine="openpyxl")
    df = pd.read_csv(uploaded_file, encoding='latin')
    return df

  
    return df
def filter_predecessors(df, id):
    predecessors = df[df['ID'] == id]['new_Predecessors'].tolist()[0]
    if predecessors is None:
        return None
    else:
        predecessors_list = [int(x) for x in predecessors.split(',') if x]
        return df[df['ID'].isin(predecessors_list)]



# Streamlit code for file upload, column selection, and ID input
uploaded_file = st.file_uploader("Choose a xlsx file", type="csv")









now = datetime.now()

if uploaded_file is not None:
    # Load data
    df = load_data(uploaded_file)
    # Rename the 'ID' column to 'ID' if it has a different name
    
     # Rename the 'ID' column to 'ID' if it has a different name
    if 'ID' not in df.columns:
        for col in df.columns:
            if 'ID' in col.upper():
                df.rename(columns={col: 'ID'}, inplace=True)
                break
        
    
    #df = pd.read_excel(uploaded_file, engine="openpyxl")

    # Column selection
    duration_column = st.selectbox('Select Duration Column', df.columns, index=df.columns.get_loc('Duration') if 'Duration' in df.columns else 0)
    start_column = st.selectbox('Select Start Date Column', df.columns, index=df.columns.get_loc('Start') if 'Start' in df.columns else 0)
    finish_column = st.selectbox('Select Finish Date Column', df.columns, index=df.columns.get_loc('Finish') if 'Finish' in df.columns else 1)
    status_column = st.selectbox('Select Status Column', df.columns, index=df.columns.get_loc('Status') if 'Status' in df.columns else 1)
    id_col = st.selectbox('Select ID Column', df.columns, index=df.columns.get_loc('ID') if 'ID' in df.columns else 1)


    # Add the prefix 'A' to the 'Value' column
    #prefix = 'A-'
    #df['ID_format'] = df['ID'].apply(lambda x: f'{prefix}{x}')
    if start_column is not None and finish_column is not None and status_column is not None and id_col is not None and duration_column is not None:

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
            
            prefix = 'ID-'
            #filtered_df['ID'] =  filtered_df['ID'].apply(lambda x: f'{prefix}{x}')
            
            
            #filtered_df['ID'] = "ID: " + filtered_df['ID'].astype(str)+ " | UID:" +filtered_df["UID"].astype(str)
            if 'UID' in filtered_df.columns:
                filtered_df['ID'] = "ID: " + filtered_df['ID'].astype(str)+ " | UID:" +filtered_df["UID"].astype(str)
            else:
                # Do something else if the conditions are not met
                # For example, you can assign a default value to the 'ID' column
                filtered_df['ID'] = "ID: " + filtered_df['ID'].astype(str)
        
            try:
                filtered_df['Days_to_Start'] = days_to_start(filtered_df,id_col, start_column, 'new_Predecessors', status_column, finish_column)  
                filtered_df[start_column] = pd.to_datetime(filtered_df[start_column])
                filtered_df[finish_column] = pd.to_datetime(filtered_df[finish_column])

                # here it fgoes
                if st.button("Gantt Static Matplot Lib"):

                    
                    def mat_plot(df, id_col, start_column, finish_column, status_column, now):
                        # Convert start and finish dates to datetime objects
                        df[start_column] = pd.to_datetime(df[start_column])
                        df[finish_column] = pd.to_datetime(df[finish_column])
                    
                        # Sort tasks by finish date and then start date
                        df = df.sort_values(by=[finish_column, start_column], ascending=False)
                    
                        # Define task colors based on status
                        status_colors = {
                            'Complete': 'green',
                            'Late': 'red',
                            'On Schedule': 'orange',
                            'Future Task': 'purple'
                        }
                    
                        # Create the Gantt chart figure
                        fig, ax = plt.subplots(figsize=(12, 6))
                    
                        # Add each task as a horizontal bar
                        for index, row in df.iterrows():
                            start_date = row[start_column]
                            end_date = row[finish_column]
                            task_id = row[id_col]
                            status = row[status_column]
                    
                            color = status_colors.get(status, 'gray')  # Default to gray if status not recognized
                            ax.barh(task_id, end_date - start_date, left=start_date, color=color, edgecolor='black', linewidth=0.5)
                    
                        # Add milestones as stars
                        zero_duration_tasks = df[df[start_column] == df[finish_column]]
                        for index, row in zero_duration_tasks.iterrows():
                            start_date = row[start_column]
                            task_id = row[id_col]
                            status = row[status_column]
                    
                            color = status_colors.get(status, 'gray')  # Default to gray if status not recognized
                            ax.scatter(start_date, task_id, marker='*', color=color, s=100, zorder=10)
                    
                        # Set chart title and labels
                        ax.set_title('Project Timeline', fontsize=16)
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Task ID')
                    
                     
                        # Add a vertical line for the current time
                        ax.axvline(x=now, color='red', linestyle='dashed', linewidth=1)
                        #ax.set_ylim(0.5, len(df) + 0.5)    
                            # Rotate x-axis labels to prevent overlapping
                        if len(df) > 20:
                            plt.xticks(rotation=45)
                        
                        # Adjust layout and return the figure
                        plt.tight_layout()
                        return fig



                    fig3 = mat_plot(filtered_df, id_col, start_column, finish_column, status_column, now)

                    # Display the figure using st.pyplot()
                    st.pyplot(fig3)
                        
                            


            
            except ValueError:

                
                st.write("Error: Could not convert one or more columns to datetime.")
                st.write("Please select appropriate start and finish date columns.")
            
            
            else:
                

                filtered_df = filtered_df.sort_values(by=[finish_column, start_column], ascending=False)
                # Define the corporate colors
                colors = {'Complete': 'green', 'Late': 'red', 'On Schedule': 'orange', 'Future Task': 'purple'}
                # Add a title to the chart
                
    
                y_labels = filtered_df['ID'].astype(str)
    
    
    
    
                # Create and display the Gantt chart
                fig = px.timeline(filtered_df, x_start=start_column, x_end=finish_column, y=id_col,
                                   color=status_column,   color_discrete_map=colors)

               

    
                
    
    
                fig.update_layout(title='Project Timeline', font=dict(family='Arial', size=16), width=800, height=600)
    
               
    
                zero_duration_tasks = filtered_df[filtered_df[start_column] == filtered_df[finish_column]]
    
                status_colors = {
                        'Future Task': '#aea7f1',
                        'In Progress': 'blue',
                        'Complete': 'limegreen'
                    }
    
    
    
                y_labels_milestones = filtered_df['ID'].astype(str)
                #fig.update_yaxes(tickmode='array', tick0=0, dtick=1)
                # Check if any task is not complete
                fig.add_trace(go.Scatter(x=zero_duration_tasks[start_column], y=zero_duration_tasks[id_col], mode='markers',
                                marker=dict(symbol='star', color=[status_colors.get(status, 'gray') for status in zero_duration_tasks[status_column]], size=15,line=dict(
                            color='black', width=1.5)), name='Milestones'))

    
    
                # Set the x-axis to display ticks every month96
                fig.update_layout(xaxis=dict(tickmode='linear', dtick='M1'))
    
    
    
    
    
                # Add a vertical line for the current time
                fig.update_layout(xaxis=dict(showgrid=False),
                                   shapes=[dict(type='line', xref='x', x0=now, x1=now, yref='paper', y0=0, y1=1,
                                               line=dict(color='red', width=1, dash='dot'))])
    
                 # Update the y-axis to have unique and linearly spaced ticks
                # Update the y-axis to have unique and linearly spaced ticks
                #fig.update_layout(yaxis=dict(tickmode='array', tickvals=y_labels, ticktext=y_labels))
    
                # Display Gantt chart and filtered DataFrame
                st.markdown('## Project Analysis')
    
                with st.container():
                    st.subheader('Gantt Chart')
                    st.plotly_chart(fig)
    
                with st.container():
                    st.subheader('Filtered DataFrame')
                    st.dataframe(filtered_df)

                def create_task_heatmap(df, value_col, actualstart_column):
                
                    #df = df[~(df["Status"].isin(["Complete"]))]
                    # Convert 'Actual Start' column to datetime type
                    df[actualstart_column] = pd.to_datetime(df[actualstart_column])
                
                    # create a new column with the month names
                    df['Month_Name'] = df[actualstart_column].dt.month_name()
                
                    # define the desired order of the month names
                    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
                
                    # set the order of the month names in the new column
                    df['Month_Name'] = pd.Categorical(df['Month_Name'], categories=month_order, ordered=True)
                
                    # reshape the DataFrame for use in the heatmap
                    heatmap_data = df.pivot_table(index='Month_Name', columns=df[actualstart_column].dt.year, values=value_col, aggfunc=lambda x: x.mean())
                
                    # set the seaborn style and color palette
                    sns.set_style('whitegrid')
                    sns.set_palette('dark')
                
                    # set the matplotlib font size and color
                    matplotlib.rcParams.update({'font.size': 12, 'text.color': '#2f4f4f'})
                
                    # create the heatmap
                    sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='.2f', cbar_kws={'format': '%.0f'})
                
                    # set the x-axis tick labels to display the years without the decimal point
                    # set the x-axis tick labels to display the years as integers
                    #plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: int(x)))
                
                    # set the plot title and axis labels
                    plt.title(f"Tasks {actualstart_column}", loc="left", size=12)
                    plt.suptitle(f'Task Progress by Month and Year ({value_col})',)
                    plt.xlabel(f'Year - Scheduled to {actualstart_column}')
                    plt.ylabel(f'Month - Scheduled to Start {actualstart_column}')
                
                    # set the size of the plot
                    plt.gcf().set_size_inches(12, 6)
                
                    # display the plot
                    plt.show()

                with st.container():
                    st.subheader('Heatmap')
                    percent_column = st.selectbox('Select % Complete Column (or a Value Column)', filtered_df.columns, index=filtered_df.columns.get_loc('Percent') if 'Percent' in filtered_df.columns else 0)
                    actual_start_column = st.selectbox('Select Actual Start Column (or a Date Column)', filtered_df.columns, index=filtered_df.columns.get_loc('Actual Start') if 'Actual Start' in filtered_df.columns else 0)

                    if percent_column is not None and actual_start_column is not None:
                        try:
                            # Remove the '%' symbol from the values in the percent_column
                            filtered_df[percent_column] = filtered_df[percent_column].str.replace('%', '', regex=False) 
                            filtered_df[percent_column] = filtered_df[percent_column].astype(int)
                            fig_s = create_task_heatmap(filtered_df, percent_column,actual_start_column)
                            #st.subheader('Heatmap')
                            #st.pyplot(fig_s)
                        except ValueError or TypeError:
                            st.write("Value error")
                        else:
                            st.set_option('deprecation.showPyplotGlobalUse', False)
                            
                            st.pyplot(fig_s)
                            
                
                    with st.container():
                        st.subheader('Filtered DataFrame')
                        st.dataframe(filtered_df)

                        










        
        else:
            st.write("No tasks found for the selected ID.")
    else:
        if st.button("Gantt chart - whole project."):
            
            try:
                    df['Days_to_Start'] = days_to_start(df,id_col, start_column, 'new_Predecessors', status_column, finish_column)  
                    df[start_column] = pd.to_datetime(df[start_column])
                    df[finish_column] = pd.to_datetime(df[finish_column])
            except ValueError:
                    st.write("Error: Could not convert one or more columns to datetime.")
                    st.write("Please select appropriate start and finish date columns.")
            else:
                
                filtered_df = df.sort_values(by=[finish_column, start_column], ascending=False)
                prefix = 'ID-'
                filtered_df['ID'] =  filtered_df['ID'].apply(lambda x: f'{prefix}{x}')
        
                # Define the corporate colors

                
                colors = {'Complete': '#a4c8e9', 'Late': '#ff6700', 'On Schedule': '#0461b2', 'Future Task': 'purple'}
                # Add a title to the chart
                
    
                y_labels = filtered_df['ID'].astype(str)
    
    
    
    
                # Create and display the Gantt chart
                fig = px.timeline(filtered_df, x_start=start_column, x_end=finish_column, y=y_labels,
                                   color=status_column,   color_discrete_map=colors)
    
    
    
    
                fig.update_layout(title='Project Timeline', font=dict(family='Arial', size=16), width=800, height=1200)
    
               
    
                zero_duration_tasks = filtered_df[filtered_df[start_column] == filtered_df[finish_column]]
    
                status_colors = {
                        'Future Task': 'purple',
                        'In Progress': 'blue',
                        'Complete': 'green',
                        'On Schedule': 'orange',
                        'Late':'red'
                    }
    
    
    
                y_labels_milestones = filtered_df['ID'].astype(str)
    
                # Check if any task is not complete
                fig.add_trace(go.Scatter(x=zero_duration_tasks[start_column], y=y_labels_milestones, mode='markers',
                                marker=dict(symbol='star', color=[status_colors.get(status, 'gray') for status in zero_duration_tasks[status_column]], size=15,line=dict(
                            color='black', width=1.5)), name='Milestones'))
    
    
                # Set the x-axis to display ticks every month
                #fig.update_layout(xaxis=dict(tickmode='linear', dtick='M1'))
    
    
    
    
    
                # Add a vertical line for the current time
                fig.update_layout(xaxis=dict(showgrid=False),
                                   shapes=[dict(type='line', xref='x', x0=now, x1=now, yref='paper', y0=0, y1=1,
                                               line=dict(color='red', width=1, dash='dot'))])
    
                 # Update the y-axis to have unique and linearly spaced ticks
            
    
                # Display Gantt chart and filtered DataFrame
                st.markdown('## Project Analysis')
    
                with st.container():
                    st.subheader('Gantt Chart')
                    st.plotly_chart(fig)
    

        
        
