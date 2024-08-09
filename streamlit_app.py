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

def aggregate_hourly(df):
    # Convert 'dt' to datetime if it's not already
    df['dt'] = pd.to_datetime(df['dt'])
    
    # Create a new column for the hourly timestamp
    df['hour'] = df['dt'].dt.floor('H')
    
    # Separate dataframes for cannabis use and mood score
    cannabis_df = df[df['event_type'] == 'cannabis_use_since_last_update_grams']
    mood_df = df[df['event_type'] == 'mood_score']
    
    # Convert 'value' to numeric, coercing errors to NaN
    cannabis_df['value'] = pd.to_numeric(cannabis_df['value'], errors='coerce')
    mood_df['value'] = pd.to_numeric(mood_df['value'], errors='coerce')
    
    # Aggregate cannabis use (sum)
    cannabis_hourly = cannabis_df.groupby('hour')['value'].sum().reset_index()
    cannabis_hourly.columns = ['dt', 'cannabis_grams']
    
    # Aggregate mood score (average)
    mood_hourly = mood_df.groupby('hour')['value'].mean().reset_index()
    mood_hourly.columns = ['dt', 'mood_score']
    
    # Merge the two aggregated dataframes
    hourly_df = pd.merge(cannabis_hourly, mood_hourly, on='dt', how='outer')
    
    # Sort by datetime
    hourly_df = hourly_df.sort_values('dt')
    
    # Replace NaN with 0 for cannabis_grams and with the overall mean for mood_score
    hourly_df['cannabis_grams'] = hourly_df['cannabis_grams'].fillna(0)
    hourly_df['mood_score'] = hourly_df['mood_score'].fillna(hourly_df['mood_score'].mean())
    
    return hourly_df

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
        
        # Aggregate data hourly
        hourly_df = aggregate_hourly(quant_df)
        
        # Display aggregated data
        st.subheader("Hourly Aggregated Data")
        st.write(hourly_df)

        # Create line charts
        st.subheader("Quantitative Metrics Over Time (Hourly)")

        # Cannabis Use Chart
        st.subheader("Cannabis Use Over Time (grams per hour)")
        st.line_chart(
            hourly_df,
            x='dt',
            y='cannabis_grams',
            x_label='Date',
            y_label='Cannabis Use (grams)',
            use_container_width=True
        )

        # Mood Score Chart
        st.subheader("Average Mood Score Over Time (per hour)")
        st.line_chart(
            hourly_df,
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