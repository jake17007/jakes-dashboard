# File: streamlit_app.py

import streamlit as st
import pandas as pd
from cannabis_mood_tracker import update_user_data, get_quantitative_metrics, initialize_firebase
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for environment variables
if 'FIREBASE_CREDENTIALS' not in os.environ:
    st.error("Firebase credentials path not found. Please set the FIREBASE_CREDENTIALS_PATH environment variable.")
    st.stop()

if 'OPENAI_API_KEY' not in os.environ:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Initialize Firebase using Streamlit's caching
db = initialize_firebase()

def aggregate_data(df, granularity):
    # Convert 'dt' to datetime if it's not already
    df['dt'] = pd.to_datetime(df['dt'])
    
    # Adjust granularity
    if granularity == 'Hourly':
        df['period'] = df['dt'].dt.floor('H')
    elif granularity == 'Daily':
        df['period'] = df['dt'].dt.floor('D')
    elif granularity == 'Weekly':
        df['period'] = df['dt'].dt.to_period('W').apply(lambda r: r.start_time)
    elif granularity == 'Minutely':
        df['period'] = df['dt'].dt.floor('T')
    
    # Separate dataframes for cannabis use and mood score
    cannabis_df = df[df['event_type'] == 'cannabis_use_since_last_update_grams']
    mood_df = df[df['event_type'] == 'mood_score']
    
    # Convert 'value' to numeric, coercing errors to NaN
    cannabis_df['value'] = pd.to_numeric(cannabis_df['value'], errors='coerce')
    mood_df['value'] = pd.to_numeric(mood_df['value'], errors='coerce')
    
    # Aggregate cannabis use (sum for each period)
    cannabis_agg = cannabis_df.groupby('period')['value'].sum().reset_index()
    cannabis_agg.columns = ['dt', 'cannabis_grams']
    
    # Aggregate mood score (average for each period)
    mood_agg = mood_df.groupby('period')['value'].mean().reset_index()
    mood_agg.columns = ['dt', 'mood_score']
    
    # Merge the two aggregated dataframes
    agg_df = pd.merge(cannabis_agg, mood_agg, on='dt', how='outer')
    
    # Sort by datetime
    agg_df = agg_df.sort_values('dt')
    
    # Replace NaN with 0 for cannabis_grams and with the overall mean for mood_score
    agg_df['cannabis_grams'] = agg_df['cannabis_grams'].fillna(0)
    agg_df['mood_score'] = agg_df['mood_score'].fillna(agg_df['mood_score'].mean())
    
    return agg_df

def main():
    st.title("Cannabis Use and Mood Tracker")

    # Hard-coded user ID (you can modify this to allow user input if needed)
    user_id = "7172032887"

    # Update data
    df = update_user_data(user_id)

    if df is not None:
        # Get quantitative metrics
        quant_df = get_quantitative_metrics(df)
        
        # Display raw data
        st.subheader("Raw Data")
        st.write(quant_df)
        
        # Period granularity selection
        st.subheader("Select Period Granularity")
        granularity = st.selectbox("Granularity", ['Minutely', 'Hourly', 'Daily', 'Weekly'])
        
        # Aggregate data based on the selected granularity
        aggregated_df = aggregate_data(quant_df, granularity)
        
        # Date range selection
        st.subheader("Select Date Range")
        min_date = st.date_input("Start date", value=aggregated_df['dt'].min().date())
        max_date = st.date_input("End date", value=aggregated_df['dt'].max().date())

        # Filter the dataframe based on the selected dates
        filtered_df = aggregated_df[(aggregated_df['dt'].dt.date >= min_date) & (aggregated_df['dt'].dt.date <= max_date)]
        
        # Display aggregated data
        st.subheader("Aggregated Data")
        st.write(filtered_df)

        # Create line charts
        st.subheader("Quantitative Metrics Over Time")

        # Cannabis Use Chart
        st.subheader(f"Cannabis Use Over Time (grams per {granularity.lower()})")
        st.line_chart(
            filtered_df,
            x='dt',
            y='cannabis_grams',
            x_label='Date',
            y_label=f'Cannabis Use (grams)',
            use_container_width=True
        )

        # Mood Score Chart
        st.subheader(f"Average Mood Score Over Time (per {granularity.lower()})")
        st.line_chart(
            filtered_df,
            x='dt',
            y='mood_score',
            x_label='Date',
            y_label='Mood Score',
            use_container_width=True
        )

    else:
        st.write("No data available for the user.")

if __name__ == "__main__":
    main()
