import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import logging

#********************************************************************* LOGGER ********************************************************************#
# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# StreamHandler to output log messages to Streamlit
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)


#******************************************************************** DATABASE *******************************************************************#
def item_list():
    data = pd.read_csv("src/Item_FullList.csv")
    return data


#**************************************************************** PLOT PREDICTION ***************************************************************#
# Function to plot the results
def plot_results(filtered_data, category, predicted_waste_train, predicted_waste_test, date_test, date_train, sigma_range_test):
    # Visualization with Plotly
    # Create an interactive plot using Plotly
    fig = go.Figure()

    # Actual waste trace
    fig.add_trace(go.Scatter(x=filtered_data['Date'], y=filtered_data[category], mode='markers', name='Actual Waste', marker=dict(color='#200fd6')))

    # Predicted Waste trace
    fig.add_trace(go.Scatter(x=date_train, y=predicted_waste_train, mode='lines', name='Predicted Waste (Train)', marker=dict(color='#f52222')))
    fig.add_trace(go.Scatter(x=date_test, y=predicted_waste_test, mode='lines', name='Predicted Waste (Test)', marker=dict(color='#00a110')))

    # Shaded uncertainty area
    fig.add_trace(go.Scatter(
        x=np.concatenate([date_test, date_test[::-1]]),
        y=np.concatenate([predicted_waste_test - 2 * sigma_range_test, (predicted_waste_test + 2 * sigma_range_test)[::-1]]),
        fill='toself',
        fillcolor='rgba(231,76,60,0.2)',  # Change the color here
        line=dict(color='rgba(255,255,255,0)'),
        name='Uncertainty'))

    fig.update_layout(
        xaxis_title='Date', 
        yaxis_title=f'{selected_category} (g)' ,
        title=f'Prediction waste for {selected_category}',
        legend=dict(x=0, y=1.1, traceorder='normal',orientation='h'),
        template= 'plotly_white'
        )

    st.plotly_chart(fig)


#*************************************************************** PAGINATION TABLE **************************************************************#
# @st.cache  # Cache the function to avoid re-execution on button click
def get_paginated_data(item_df, current_page, items_per_page):
    if items_per_page == "All":
        return item_df
    else:
        start_idx = (current_page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(item_df))
        return item_df.iloc[start_idx:end_idx, :]


#****************************************************************** ITEMS TAB *****************************************************************#
# Function to display item list table
def display_item_list_table(csv_path):
    try:
        item_df = pd.read_csv(csv_path)

        # Show the total data
        Total_data = len(item_df)
        st.subheader("This is an overview of all items, along with their details. ",help=f"Total data stored : {Total_data}")

        desired_column_order = ['Date', 'Item Code', 'Item Description', 'Carbohydrates', 'Fiber', 'Protein', 'Fat', 'Quantity', 'Unit Price']

        act_waste_per_item = ['Item Description','Carbohydrates (g)','Fiber (g)','Protein (g)','Fat (g)']

        # initialize new desired columns for dataset
        rearranged_df = item_df[desired_column_order]

        act_waste = item_df[act_waste_per_item]

        # Initialize session state if it doesn't exist
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 1

        # Number of items per page
        items_per_page = st.sidebar.selectbox("Items per page", [5, 10, 20, 50, 100, "All"], index=5)

         # Checkbox to reset the date filter
        reset_filter = st.sidebar.checkbox("Reset Date Filter", value=True, help="Uncheck it to enable the filter date.")

        # Filter the date to display
        filter_date = st.sidebar.date_input("Please choose the date to filter the items list", min_value=datetime(2023, 1, 1), max_value=datetime(2023, 9, 30), value=datetime(2023, 9, 30), format="DD/MM/YYYY", disabled=reset_filter) 
        
        # Sidebar for search input
        Total_item = len(item_df['Item Description'].unique())
        selected_item = st.sidebar.selectbox('Search Items:', item_df['Item Description'].unique(), index=None, placeholder="Choose item...", help=f"There are {Total_item} items available")

        # Define a function to get dummy item details with all values set to 0
        def get_dummy_item_details():
            dummy_item_details = {
                'Carbohydrates (g)': 0,
                'Fiber (g)': 0,
                'Protein (g)': 0,
                'Fat (g)': 0
            }
            return dummy_item_details

        # Filter items based on selected item
        if selected_item:
            selected_row = act_waste[act_waste['Item Description'] == selected_item].iloc[0]
        else:
            # If selected item is None, use dummy item details
            selected_row = get_dummy_item_details()

        # Display selected item details in the sidebar
        st.sidebar.subheader("Nutrition per Item Details:")

        for col_name, col_value in selected_row.items():
            st.sidebar.markdown(
                f"""
                <div style='display: flex; margin-bottom: 10px;'>
                    <div style='border: 1px solid #09111a; padding: 10px; flex: 0.4; border-radius: 10px; 
                    background-color: #09111a; color: white;'><strong>{col_name}:</strong></div>
                    <div style='border: 1px solid #09111a; padding: 10px; margin-left: 10px; flex: 0.6; border-radius: 10px; 
                    background-color: #09111a; color: white;'>{col_value}</div>
                </div>
                """, 
                unsafe_allow_html=True
            )

        # Apply the date filter if a date is selected and the reset checkbox is not checked
        if filter_date is not None and not reset_filter:
            # Convert the DataFrame's "Date" column to match the format of the date input
            rearranged_df['Date'] = pd.to_datetime(rearranged_df['Date'], format="%d/%m/%Y").dt.strftime("%d/%m/%Y")
            rearranged_df = rearranged_df[rearranged_df['Date'] == filter_date.strftime("%d/%m/%Y")]
                
            if rearranged_df.empty:
                st.error("No data available for the selected date.")

            # Show the total of item for selected date
            Total_filtered = len(rearranged_df)
            date = filter_date.strftime("%d/%m/%Y")
            st.write(f"The amount of item(s) on {date} : {Total_filtered}") 
        

        # Multiselect widget for column selection
        selected_columns = st.multiselect("Select Columns to Display", rearranged_df.columns)

        # Display the DataFrame with selected columns
        if items_per_page == "All":
            st.dataframe(rearranged_df[selected_columns], hide_index=True)
        else:
            paginated_data = get_paginated_data(rearranged_df[selected_columns], st.session_state.current_page, items_per_page)
            st.dataframe(paginated_data, hide_index=True)

        # Add arrows at the bottom for pagination controls
        col0, col1, col2, col3 = st.columns([0.6, 0.2, 0.17, 0.2])
        if col1.button("&#9664; Previous"):
            st.session_state.current_page = max(st.session_state.current_page - 1, 1)

        total_pages = 1 if items_per_page == "All" or len(item_df) == 0 else max(1, (len(item_df) + items_per_page - 1) // items_per_page)
        col2.write(f"Page {st.session_state.current_page} of {total_pages}")

        if col3.button("Next &#9654;"):
            st.session_state.current_page = min(st.session_state.current_page + 1, total_pages)

    except FileNotFoundError:
        st.error("Item list CSV file not found. Please make sure the file exists.")


#****************************************************************** LOAD MODEL *****************************************************************#
@st.cache_data
def load_model(category):
    try:
        model_filename = f"savedModel/{category}/{category} model.pkl"
        loaded_model = joblib.load(model_filename)
        st.success("Model loaded successfully!")
        return loaded_model
    except FileNotFoundError:
        st.info(f"The food waste category of {category} does not have a saved model. Sorry for the inconvenience; we will train the model as soon as possible. Please select a different category.")
        return None
    

#**************************************************************** PREDICTION TAB ***************************************************************#
# Function for prediction food waste
def  main(category):
    # Dummy figure
    dummy_fig = go.Figure()
    dummy_date = pd.to_datetime('2023-01-01')
    dummy_fig.add_trace(go.Scatter(x=[dummy_date, dummy_date], y=[0, 0], mode='markers', name='Actual Waste'))
    dummy_fig.add_trace(go.Scatter(x=[dummy_date, dummy_date], y=[0, 0], mode='lines', name='Predicted Waste',
                                line=dict(color='#e74c3c')))
    dummy_fig.update_layout(
        xaxis=dict(title='Date'),
        yaxis=dict(title='Waste'),
        title='Select a category to view the graph',
        showlegend=True,
    )

    if category is not None:
        # Read data
        df = pd.read_csv('src/Weekly_Average_FoodWaste.csv')

        # Convert date column to datetime format
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Define the chosen category for plotting
        chosen_category = category

        # Filter the data for the chosen category
        selected_columns = ['Date', chosen_category]
        filtered_data = df[selected_columns]
        # logger.info(filtered_data)

        split_ratio = 0.97
        split_index = int(len(filtered_data) * split_ratio)
        df_train = filtered_data[:split_index + 1]
        df_test = filtered_data[split_index:]

        # Extract features and target for the current state
        start = filtered_data['Date'].min()
        end = filtered_data['Date'].max()
        range_datetime = (end - start).days

        # Normalize date and waste variables
        reference_date = datetime(2023, 1, 1)
        normalized_date = (df_train['Date'] - reference_date).dt.days.values.reshape(-1, 1) / range_datetime
        normalized_waste = df_train[selected_category].values.reshape(-1, 1) / np.max(filtered_data[selected_category])

        X = normalized_date
        y = normalized_waste

        # Load the saved GP model
        loaded_model = load_model(chosen_category)

        # Make predictions when the model is loaded
        if loaded_model is not None:
            # prediction for train data
            start_date = df_train['Date'].min()
            end_date =   df_train['Date'].max() 
            date_train = pd.date_range(start=start_date, end=end_date, freq='D')

            # Normalize the date range
            normalized_date_train = (date_train - reference_date).days / range_datetime
            X_train = normalized_date_train.values.reshape(-1, 1)

            # Make predictions for the date range using the GP model
            y_pred_train, sigma_range_train = loaded_model.predict(X_train, return_std=True)

            # Denormalize the predicted wastes
            predicted_waste_train = y_pred_train * np.max(filtered_data[chosen_category])

            # Options for the selectbox
            options = [1, 5, 10, 20, 30]

            # Selectbox with 30 as the default value
            adv_pred = st.selectbox("Day of Advanced Forecast", options, index=options.index(30))

            # predict for test data
            start_dates = df_test['Date'].min()
            end_dates =   df_test['Date'].max()+ timedelta(days=adv_pred)
            date_test = pd.date_range(start=start_dates, end=end_dates, freq='D')

            # Normalize the date range
            normalized_date_test = (date_test - reference_date).days / range_datetime
            X_test = normalized_date_test.values.reshape(-1, 1)

            # Make predictions for the date range using the GP model
            y_pred_test, sigma_range_test = loaded_model.predict(X_test, return_std=True)

            # Denormalize the predicted wastes
            predicted_waste_test = y_pred_test * np.max(filtered_data[chosen_category])

            # Get the last date and its corresponding waste value
            last_predicted_date = date_test[-1]
            formatted_last_predicted_date = last_predicted_date.strftime("%d %B %Y")
            last_predicted_waste = round(predicted_waste_test[-1],2)

            # Plot results
            plot_results(filtered_data, category, predicted_waste_train, predicted_waste_test, date_test, date_train, sigma_range_test)

            st.text('INFO', help=f'The date represents the weekly average of {chosen_category} waste, with Sunday serving as the reference week.')

            st.write(f"The amount of {chosen_category} that is predicted to be wasted in the next 30 days after September on {formatted_last_predicted_date} is {last_predicted_waste}g")

            st.markdown("---")
        
            top_wasted_items = (
                item_list().groupby("Item Description")[chosen_category].sum()
                .reset_index()
                .nlargest(10, columns=chosen_category)
            )

            # Sort the DataFrame by the chosen category in descending order
            top_wasted_items_sorted = top_wasted_items.sort_values(by=chosen_category, ascending=True)

            # Get the maximum and minimum values
            max_value = top_wasted_items_sorted[chosen_category].max()
            min_value = top_wasted_items_sorted[chosen_category].min()

            item_description_max = top_wasted_items_sorted.loc[top_wasted_items_sorted[chosen_category].idxmax(), "Item Description"]
            item_description_min = top_wasted_items_sorted.loc[top_wasted_items_sorted[chosen_category].idxmin(), "Item Description"]

            # Display the horizontal bar chart
            fig_waste_items = px.bar(
                top_wasted_items_sorted,
                x=chosen_category,
                y="Item Description",  # Set "Item Description" as the y-axis
                orientation='h',  # Specify the orientation as horizontal
                title=f'<b>Top Wasted Items by {chosen_category} in grams (g)</b>',  
                color_discrete_sequence=["#0083B8"] * len(top_wasted_items_sorted),
                template="plotly_white",
            )

            st.plotly_chart(fig_waste_items)

            st.write(f"The Highest Wasted Items in bar chart : {item_description_max}  ({max_value}g)")
            st.write(f"The Lowest Wasted Items in bar chart  : {item_description_min}  ({min_value}g)")

        else:
            # Display the dummy figure if the model is not loaded
            st.plotly_chart(dummy_fig)
    else:
        # Display the dummy figure if the model is not loaded
            st.plotly_chart(dummy_fig)


#******************************************************************* HOME TAB ******************************************************************#
# Display the pie chart 
def display_pie_topwasted(selected_columns):
    st.write(f"The top items in terms of {selected_columns.lower()} that have been wasted all this time.")
    top_n = st.slider("How many top items do you want to display?", 1, 20, 10)
    top_wasted_items = (
        item_list().groupby("Item Description")[selected_columns].sum()
        .reset_index()
        .nlargest(top_n, columns=selected_columns)
    )

    if selected_columns == 'Quantity':
        # Display the top wasted items in a pie chart
        fig = px.pie(top_wasted_items, values=selected_columns, names='Item Description', title='Quantity of wasted items')
        st.plotly_chart(fig)
    else:
        # Display the top wasted items in a pie chart
        fig = px.pie(top_wasted_items, values=selected_columns, names='Item Description', title='Price of wasted items')
        st.plotly_chart(fig)

    # Get the maximum and minimum values
    max_value = top_wasted_items[selected_columns].max()
    min_value = top_wasted_items[selected_columns].min()

    item_description_max = top_wasted_items.loc[top_wasted_items[selected_columns].idxmax(), "Item Description"]
    item_description_min = top_wasted_items.loc[top_wasted_items[selected_columns].idxmin(), "Item Description"]
    
    if selected_columns == 'Quantity':
        st.write(f"The Highest Wasted Items in pie chart : {item_description_max}  ({max_value} units)")
        st.write(f"The Lowest Wasted Items in pie chart  : {item_description_min}  ({min_value} units)")
    else:
        st.write(f"The Highest Wasted Items in pie chart : {item_description_max}  (RM{max_value})")
        st.write(f"The Lowest Wasted Items in pie chart  : {item_description_min}  (RM{min_value})")


# Display the pie chart for the selected month
def display_pie_topwasted_by_month(selected_columns):
    # Read data
    df = pd.read_csv('src/Item_Quantity.csv')

    st.write(f"The top items in terms of {selected_columns.lower()} that have been wasted by month.")

    # Get the top wasted items for the selected month
    top_n = st.slider("How many top items do you want to display?", 1, 20, 10)

    months = df['Month'].unique()

    # Allow the user to select a month
    selected_month = st.selectbox("Please choose which month do you want to display", months, help="This months is in the year 2023.")
    st.text("Last updated on September 2023")

    # Filter the data for the selected month
    filtered_df = df[df['Month'] == selected_month]

    # Group by Item Description and compute the sum of quantities
    top_wasted_items = (
        filtered_df.groupby("Item Description")[selected_columns].sum()
        .reset_index()
        .nlargest(top_n, columns=selected_columns)
    )

    if selected_columns == 'Quantity':
        # Display the top wasted items in a pie chart
        fig = px.pie(top_wasted_items, values=selected_columns, names='Item Description', title='Quantity of wasted items')
        st.plotly_chart(fig)
    else:
        # Display the top wasted items in a pie chart
        fig = px.pie(top_wasted_items, values=selected_columns, names='Item Description', title='Price of wasted items')
        st.plotly_chart(fig)

    # Get the maximum and minimum values
    max_value = top_wasted_items[selected_columns].max()
    min_value = top_wasted_items[selected_columns].min()

    item_description_max = top_wasted_items.loc[top_wasted_items[selected_columns].idxmax(), "Item Description"]
    item_description_min = top_wasted_items.loc[top_wasted_items[selected_columns].idxmin(), "Item Description"]

    if selected_columns == 'Quantity':
        st.write(f"The Highest Wasted Items in pie chart : {item_description_max}  ({max_value} units)")
        st.write(f"The Lowest Wasted Items in pie chart  : {item_description_min}  ({min_value} units)")
    else:
        st.write(f"The Highest Wasted Items in pie chart : {item_description_max}  (RM{max_value})")
        st.write(f"The Lowest Wasted Items in pie chart  : {item_description_min}  (RM{min_value})")

#**************************************************************** DASHBOARD MENU ***************************************************************#
# Display the dashboard name
col1, col2 = st.columns([0.2,1])

with col1:
    st.image("src/WasteLess.png", width=500)

# horizontal menu
selected = option_menu(
    menu_title= None, #required
    options=["Home","Prediction","Items"],
    icons=["house","bar-chart-line","file-earmark-fill"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

# Add navigation bar with buttons
if selected == "Home":
    st.markdown("<h1 style='color: green; font-size: 55px; line-height: 0.8;'>Reduce Waste</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='color: green; font-size: 22px; line-height: 1.2;'>Welcome to our website devoted to the fight against food waste problem!</h1>", unsafe_allow_html=True)
    st.markdown("""
                
                    Food waste has become a serious concern in today's globe, as the population is growing at an alarming rate. Artificial intelligence (AI) has the potential to significantly reduce food waste. AI can be used to optimise supply chain management, predict demand, and reduce food waste. At WasteLess, we are enthusiastic about decreasing food waste and building a more sustainable future. Our objective is to raise awareness about the unforeseen amount of food waste by creating a prediction model that forecasts food waste using machine learning techniques.

                    Join us in the fight against food waste and help us make a difference. Let us work together to save resources, preserve food, and build a more sustainable future for future generations.

                    Start making a difference today. We can minimise food waste together, one step at a time!
                """)
    st.markdown("---")
    st.subheader("Top Wasted Items - Stock Analysis")
    # Get the top wasted items
    selected_columns = st.selectbox("Select a criteria to display", ['Quantity', 'Total Price'],index=0)
    
    display_pie_topwasted(selected_columns)
    st.markdown("---")
    display_pie_topwasted_by_month(selected_columns)

if selected == "Prediction":
    # Define the chosen category for plotting
    st.title(f"Food Waste {selected} by Nutrition", help="This prediction model employs the Gaussian Process Regressor")
    st.subheader("The data visualization displays the food waste trends of the selected nutrition category.")
    selected_category = st.selectbox("Select Food Waste Category", ['Carbohydrates', 'Protein', 'Fat', 'Fiber', 'Vitamin'],index=None, placeholder="Choose a category...")
    main(selected_category)

if selected == "Items":
    st.title("Item List")
    # Specify the path to your item list CSV file
    item_list = "src/Item_FullList.csv"
    display_item_list_table(item_list)
    

    
