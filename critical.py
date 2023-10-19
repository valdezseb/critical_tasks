import databutton as db


user = db.user.get()
name = user.name if user.name else "you"

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import math
import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.markers import MarkerStyle
import numpy as np


from pandas.tseries.offsets import CustomBusinessDay

def display_modified_gantt_chart(filtered_df_selected, id_col, start_column, finish_column, status_column, now):
    # Sort the DataFrame by finish date and start date
    filtered_df = filtered_df_selected.sort_values(by=[finish_column, start_column], ascending=False)
    
    # Define the corporate colors for the Gantt chart
    colors = {'Complete': 'green', 'Late': 'red', 'On Schedule': 'orange', 'Future Task': 'purple'}
    
    # Create and display the Gantt chart
    fig_modified = px.timeline(filtered_df, x_start=start_column, x_end=finish_column, y=id_col,
                               color=status_column, color_discrete_map=colors)
    fig_modified.update_layout(title='Modified Project Timeline', font=dict(family='Arial', size=16),
                               width=800, height=600)
    
    # Add milestones as scatter markers
    zero_duration_tasks_modified = filtered_df[filtered_df[start_column] == filtered_df[finish_column]]
    status_colors = {
        'Future Task': '#aea7f1',
        'In Progress': 'blue',
        'Complete': 'limegreen'
    }
    fig_modified.add_trace(go.Scatter(x=zero_duration_tasks_modified[start_column], y=zero_duration_tasks_modified[id_col],
                                     mode='markers', marker=dict(symbol='star', color=[
                                     status_colors.get(status, 'gray') for status in zero_duration_tasks_modified[status_column]],
                                     size=15, line=dict(color='black', width=1.5)), name='Milestones'))
    
    # Set the x-axis tick format to display ticks every month
    fig_modified.update_layout(xaxis=dict(tickmode='linear', dtick='M1'))
    
    # Add a vertical line for the current time
    fig_modified.update_layout(xaxis=dict(showgrid=False),
                               shapes=[dict(type='line', xref='x', x0=now, x1=now, yref='paper', y0=0, y1=1,
                                            line=dict(color='red', width=1, dash='dot'))])
    
    # Display the modified Gantt chart
    st.subheader('Modified Gantt Chart')
    st.plotly_chart(fig_modified)


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


def calculate_days_to_start():
    try:
        # Get the task details from the user
        task_id = st.text_input("Enter the task ID:")
        start_date = st.text_input("Enter the task start date (MM/DD/YYYY):")
        predecessors = st.text_input("Enter the IDs of the predecessors (comma-separated):")

        # Convert start date to datetime object
        start_date = datetime.strptime(start_date, "%m/%d/%Y")

        # Get the latest finish date of the predecessors
        latest_finish_date = None
        if predecessors:
            pred_finish_dates = []
            for pred_id in predecessors.split(","):
                pred_finish_date = st.text_input(f"Enter the finish date for task {pred_id} (MM/DD/YYYY):")
                pred_finish_date = datetime.strptime(pred_finish_date, "%m/%d/%Y")
                pred_finish_dates.append(pred_finish_date)

            latest_finish_date = max(pred_finish_dates)

        # Calculate the number of working days to start
        if latest_finish_date:
            # Create a custom business day calendar
            bus_day_cal = CustomBusinessDay(weekmask="Mon Tue Wed Thu Fri")
            # Calculate the number of business days between start date and latest finish date
            num_working_days = len(pd.bdate_range(start=start_date, end=latest_finish_date, freq=bus_day_cal))
        else:
            num_working_days = 0

        # Display the result
        st.write(f"The number of working days to start task {task_id} is: {num_working_days} days")

        # Plot the Gantt chart
        if st.button("Show Gantt Chart"):
            # Check if there are any predecessors
            if predecessors:
                predecessor_list = [int(p) for p in predecessors.split(",") if p]
                filtered_df = df[df["ID"].isin(predecessor_list)]
            else:
                filtered_df = pd.DataFrame()

            # Create the Gantt chart
            fig, ax = plt.subplots(figsize=(12, 6))

            # Set the x-axis to datetime format
            ax.xaxis_date()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
            ax.yaxis.set_tick_params(labelsize=8)

            # Plot the task as a horizontal bar
            ax.barh(task_id, 1, left=start_date, color='skyblue', alpha=0.8)

            # Plot the predecessor tasks as horizontal bars
            for pred_id in predecessor_list:
                pred_start_date_str = filtered_df[filtered_df["ID"] == pred_id][start_column].values[0]
                pred_start_date = datetime.strptime(pred_start_date_str, "%m/%d/%Y")  # Convert to datetime object
                ax.barh(pred_id, 1, left=pred_start_date, color='lightgray')
                # Set the y-axis label
                ax.set_ylabel('Task ID')

            # Set the chart title
            ax.set_title('Gantt Chart')

            # Fit the x-axis labels within the figure width
            fig.autofmt_xdate()

            # Remove y-axis ticks and spines
            ax.yaxis.set_ticks([])
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Show the plot
            st.pyplot(fig)

    except ValueError as e:
        st.error(str(e))





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
                    heatmap_data = df.pivot_table(index='Month_Name', columns=df[actualstart_column].dt.year, values=value_col, aggfunc= lambda x: x.mean())
                
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
                    plt.ylabel(f'Month - Scheduled to {actualstart_column}')
                
                    # set the size of the plot
                    plt.gcf().set_size_inches(12, 6)
                
                    # display the plot
                    plt.show()






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

#@st.cache_data
def days_to_start(df, id_col, start_col, pred_col, status_col, finish_col):
    df[start_col] = pd.to_datetime(df[start_col])
    df[finish_col] = pd.to_datetime(df[finish_col])
    last_finish_dates = {}

    for index, row in df.iterrows():
        task_id = row[id_col]
        pred_ids = row[pred_col]

        if pd.isna(pred_ids):
            df.at[index, 'Days_to_Start'] = 0
        else:
            pred_finish_dates = []
            for pred_id in pred_ids.split(','):
                if pred_id:
                    pred_df = df[(df[id_col] == int(pred_id)) & (df[status_col] != "Complete")].sort_values(by=finish_col, ascending=False)
                    if not pred_df.empty:
                        pred_finish_date = pred_df[finish_col].values[0]
                        pred_finish_dates.append(pred_finish_date)

            if pred_finish_dates:
                latest_finish_date = max(pred_finish_dates)
                task_start_date = row[start_col]

                days_to_start = (latest_finish_date - task_start_date).days

                df.at[index, 'Days_to_Start'] = days_to_start
            else:
                df.at[index, 'Days_to_Start'] = 0

        last_finish_dates[task_id] = row[finish_col]

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
        


    # Define the available columns
    available_columns = df.columns.tolist()

    # Create the form for column selection
    with st.form("column_select_form"):
      # Add the column selection fields to the form
        duration_column = st.selectbox('Select Duration Column', options=available_columns, index=available_columns.index('TaskDuration') if 'TaskDuration' in available_columns else 0)

        start_column = st.selectbox('Select Start Date Column', options=available_columns, index=available_columns.index('Start') if 'Start' in available_columns else 0)
        finish_column = st.selectbox('Select Finish Date Column', options=available_columns, index=available_columns.index('Finish') if 'Finish' in available_columns else 0)
        status_column = st.selectbox('Select Status Column', options=available_columns, index=available_columns.index('Status') if 'Status' in available_columns else 0)
        id_col = st.selectbox('Select ID Column', options=available_columns, index=available_columns.index('ID') if 'ID' in available_columns else 0)
        # Submit button to submit the form
        column_select_submit = st.form_submit_button("Submit")

    

    # Add the prefix 'A' to the 'Value' column
    #prefix = 'A-'
    #df['ID_format'] = df['ID'].apply(lambda x: f'{prefix}{x}')
    # Only process the data if the form is submitted

    if column_select_submit:
        if start_column is not None and finish_column is not None and status_column is not None and id_col is not None and duration_column is not None:

            #df['New_Duration'] = df[duration_column].apply(duration_to_days)
             st.success("All columns filled.")   
            #df['new_Predecessors'] = df['Predecessors'].apply(extract_number)    
            
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

        df['new_Predecessors'] = df['Predecessors'].apply(extract_number)
        
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
                 
                filtered_df[start_column] = pd.to_datetime(filtered_df[start_column])
                filtered_df[finish_column] = pd.to_datetime(filtered_df[finish_column])
                filtered_df['Days_to_Start'] = days_to_start(filtered_df,id_col, start_column, 'new_Predecessors', status_column, finish_column)
                # here it fgoes
                if st.button("Gantt Static Matplot Lib"):

                    
                    



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
                    
        
                    
                        
                with st.container():
                    st.subheader('Heatmap')
                    percent_column = st.selectbox('Select % Complete Column (or a Value Column)', filtered_df.columns, index=filtered_df.columns.get_loc('Percent') if 'Percent' in filtered_df.columns else 0)
                    actual_start_column = st.selectbox('Select Actual Start Column (or a Date Column)', filtered_df.columns, index=filtered_df.columns.get_loc('Actual Start') if 'Actual Start' in filtered_df.columns else 0)
                    #mean_x = lambda x: x.mean()
                    #aggfunction = st.selectbox('Select Agg Function (sum, mean)', ["sum", " 'lambda x: x.mean()'"], index=0)
                    
                    if percent_column is not None and actual_start_column is not None :
                        try:
                            # Remove the '%' symbol from the values in the percent_column
                            filtered_df[percent_column] = filtered_df[percent_column].str.replace('%', '', regex=False) 
                            filtered_df[percent_column] = filtered_df[percent_column].astype(int)
                            fig_s = create_task_heatmap(filtered_df, percent_column,actual_start_column)
                            #st.subheader('Heatmap')
                            #st.pyplot(fig_s)
                        except (ValueError, TypeError, AttributeError):
                            st.write("Value error")
                        else:
                            st.set_option('deprecation.showPyplotGlobalUse', False)
                            
                            st.pyplot(fig_s)
                            
                
                    with st.container():
                        st.subheader('Predecessors Finish Dates')
                        #st.dataframe(filtered_df)
                        
                
                    
                    # Allow the user to edit the finish dates dynamically
                        # Get the list of unique task IDs
                        try:
                            task_ids = filtered_df['ID'].unique().tolist()
                        except TypeError:
                            st.warning("No Predecessors found!")
                        else:
                            # Ask the user to select the task IDs to update the finish dates
                            selected_task_ids = st.multiselect("Select Task IDs to Update Finish Dates", task_ids)
                    
                            # Filter the DataFrame based on the selected task IDs
                            filtered_df_selected = filtered_df[filtered_df['ID'].isin(selected_task_ids)]
  
                            # Display the filtered DataFrame
                            st.dataframe(filtered_df_selected)
                    
                            # Ask the user if they want to modify the finish date column for the selected tasks
                            update_finish_date = st.checkbox("Update Finish Dates for Selected Tasks")
                    
                            if update_finish_date:
                                # Allow the user to edit the finish dates dynamically for the selected tasks
                                for task_id in selected_task_ids:
                                    finish_date = st.date_input(f"Edit Finish Date for Task {task_id}",
                                                                filtered_df_selected.loc[filtered_df_selected['ID'] == task_id, finish_column].iloc[0],
                                                                key=str(task_id))
                                    filtered_df_selected.loc[filtered_df_selected['ID'] == task_id, finish_column] = finish_date
                    
                                # Display the updated DataFrame with the modified finish dates
                                st.subheader("Updated DataFrame - Finish Dates")
                                st.dataframe(filtered_df_selected)

            
                                display_modified_gantt_chart(filtered_df_selected, id_col, start_column, finish_column, status_column, now)


                                                                 
        else:
            st.write("No tasks found for the selected ID.")
    else:
        if st.button("Gantt chart - whole project."):
            
            try:
                    #df['Days_to_Start'] = days_to_start(df,id_col, start_column, 'new_Predecessors', status_column, finish_column)  
                    df[start_column] = pd.to_datetime(df[start_column])
                    df[finish_column] = pd.to_datetime(df[finish_column])
            except (ValueError, TypeError):
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



        
        
