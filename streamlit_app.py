import streamlit as st
import pandas as pd
from cannabis_mood_tracker import update_user_data, get_quantitative_metrics, initialize_firebase
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import altair as alt

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
        df['period'] = df['dt'].to_period('W').apply(lambda r: r.start_time)
    elif granularity == 'Minutely':
        df['period'] = df['dt'].dt.floor('T')
    
    # Separate dataframes for different metrics
    metrics = {
        'cannabis_use_since_last_update_grams': 'cannabis_grams',
        'mood_score': 'mood_score',
        'monthly_cashflow_income': 'monthly_cashflow_income',
        'monthly_cashflow_expenses': 'monthly_cashflow_expenses',
        'monthly_cashflow_savings': 'monthly_cashflow_savings',
        'monthly_cashflow_savings_rate': 'monthly_cashflow_savings_rate'
    }
    
    aggregated_dfs = []

    for event_type, column_name in metrics.items():
        metric_df = df[df['event_type'] == event_type]
        
        if not metric_df.empty:
            metric_df['value'] = pd.to_numeric(metric_df['value'], errors='coerce')
            
            if event_type == 'cannabis_use_since_last_update_grams':
                agg_df = metric_df.groupby('period')['value'].sum().reset_index()
            elif event_type == 'mood_score':
                agg_df = metric_df.groupby('period')['value'].mean().reset_index()
            else:
                agg_df = metric_df.groupby('period').last().reset_index()
            
            # Ensure we only have 'period' and 'value' columns
            agg_df = agg_df[['period', 'value']]
            agg_df.columns = ['dt', column_name]
            aggregated_dfs.append(agg_df)
        else:
            # If no data for this metric, create an empty DataFrame with the correct columns
            agg_df = pd.DataFrame(columns=['dt', column_name])
            aggregated_dfs.append(agg_df)
    
    # Merge all aggregated dataframes
    final_df = aggregated_dfs[0]
    for df in aggregated_dfs[1:]:
        final_df = pd.merge(final_df, df, on='dt', how='outer')
    
    # Sort by datetime
    final_df = final_df.sort_values('dt')
    
    # Replace NaN with appropriate values
    final_df['cannabis_grams'] = final_df['cannabis_grams'].fillna(0)
    final_df['mood_score'] = final_df['mood_score'].fillna(final_df['mood_score'].mean())
    
    # For financial metrics, forward fill (use the last known value)
    financial_columns = ['monthly_cashflow_income', 'monthly_cashflow_expenses', 
                         'monthly_cashflow_savings', 'monthly_cashflow_savings_rate']
    final_df[financial_columns] = final_df[financial_columns].ffill()
    
    return final_df

def safe_plot(df, x, y, title):
    if y in df.columns:
        # Make sure NaN values are preserved to create gaps
        chart_data = df[[x, y]].copy()
        chart_data['dt'] = pd.to_datetime(chart_data['dt'])

        # Use Altair to create a line chart with points and rotated x-axis labels
        chart = alt.Chart(chart_data).mark_line(point=True).encode(
            x=alt.X(x, title='Date', type='temporal', axis=alt.Axis(format='%Y-%m-%d %H:%M', labelAngle=-45)),
            y=alt.Y(y, title=y.replace('_', ' ').title())
        ).properties(
            title=title,
            width='container',
            height=400
        )
        
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning(f"Column {y} not found in the data")

def main():
    st.title("Jake's Dashboard")

    # Hard-coded user ID (you can modify this to allow user input if needed)
    user_id = "7172032887"

    # Update data
    df = update_user_data(user_id)

    if df is not None:
        # Get quantitative metrics
        quant_df = get_quantitative_metrics(df)
        
        # Period granularity selection (default to Hourly)
        st.subheader("Select Period Granularity")
        granularity = st.selectbox("Granularity", ['Hourly', 'Minutely', 'Daily', 'Weekly'], index=0)
        
        # Aggregate data based on the selected granularity
        aggregated_df = aggregate_data(quant_df, granularity)
        
        # Date range selection (default to past 7 days)
        st.subheader("Select Date Range")
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=7)
        min_date = st.date_input("Start date", value=start_date, max_value=end_date)
        max_date = st.date_input("End date", value=end_date, min_value=min_date, max_value=end_date)

        # Filter the dataframe based on the selected dates
        filtered_df = aggregated_df[(aggregated_df['dt'].dt.date >= min_date) & (aggregated_df['dt'].dt.date <= max_date)]

        # Create line charts
        safe_plot(filtered_df, 'dt', 'cannabis_grams', f"Cannabis Use Over Time (grams per {granularity.lower()})")
        safe_plot(filtered_df, 'dt', 'mood_score', f"Average Mood Score Over Time (per {granularity.lower()})")
        
        st.subheader("Financial Metrics")
        financial_metrics = ['monthly_cashflow_income', 'monthly_cashflow_expenses', 'monthly_cashflow_savings', 'monthly_cashflow_savings_rate']
        for metric in financial_metrics:
            safe_plot(filtered_df, 'dt', metric, metric.replace('_', ' ').title())

        # Display raw data
        st.subheader("Raw Data")
        st.write(quant_df)

        # Display aggregated data
        st.subheader("Aggregated Data")
        st.write(filtered_df)

    else:
        st.write("No data available for the user.")

if __name__ == "__main__":
    main()
