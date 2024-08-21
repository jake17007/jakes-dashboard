import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import pandas as pd
from datetime import datetime
from openai import OpenAI
import os
import json
import streamlit as st

# OpenAI API key
api_key = os.getenv('OPENAI_API_KEY')

@st.cache_resource
def initialize_firebase():
    if not firebase_admin._apps:
        # Get the Firebase credentials JSON string from an environment variable
        cred_json = os.getenv('FIREBASE_CREDENTIALS')
        print('ran')
        if cred_json:
            cred_dict = json.loads(cred_json, strict=False)
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
        else:
            raise ValueError("Firebase credentials not found in environment variables")
    return firestore.client()

class CannabisMetricAgent:
    def __init__(self):
        self.brain = OpenAI(api_key=api_key)
        self.role = "Cannabis Use Metric Interpreter"

    def process_metric(self, metric_value):
        print(f"Processing cannabis metric: {metric_value}")
        prompt = f"""
        Task: Convert the given cannabis use amount to a numerical value in grams.

        Input: {metric_value}

        Please analyze the input and return a single float number representing the amount in grams. 
        Use the following guidelines:
        - If the amount is already in grams, convert it to a float.
        - If no use is reported (e.g., "0"), return 0.
        - For qualitative descriptions, estimate a reasonable amount in grams.
        - If an edible with milligrams is specified, use the following conversion: 10 mg gummy == 0.5 g cannabis.

        Return only the numerical value as a float, with no additional text or explanation.
        """

        print("Sending request to OpenAI API for cannabis metric...")
        chat = self.brain.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": self.role},
                {"role": "user", "content": prompt}
            ]
        )

        response = chat.choices[0].message.content.strip()
        print(f"Received response from OpenAI API for cannabis metric: {response}")
        
        try:
            result = float(response)
            print(f"Converted cannabis metric to: {result} grams")
        except ValueError:
            result = 0.0
            print(f"Warning: Couldn't convert '{response}' to a float. Defaulting to 0.0 grams.")

        return result

class MoodMetricAgent:
    def __init__(self):
        self.brain = OpenAI(api_key=api_key)
        self.role = "Mood Metric Interpreter"

    def process_metric(self, metric_value):
        print(f"Processing mood metric: {metric_value}")
        prompt = f"""
        Task: Convert the given mood description to a numerical score from 1 to 5.

        Input: {metric_value}

        Please analyze the input and return a single integer representing the mood score. 
        Use the following guidelines:
        - 1 represents the lowest mood (very bad, terrible, etc.)
        - 5 represents the highest mood (excellent, amazing, etc.)
        - Use 2, 3, and 4 for intermediate moods
        - If the input is already a number between 1 and 5, return that number
        - For qualitative descriptions, estimate the most appropriate score

        Return only the numerical value as an integer from 1 to 5, with no additional text or explanation.
        """

        print("Sending request to OpenAI API for mood metric...")
        chat = self.brain.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": self.role},
                {"role": "user", "content": prompt}
            ]
        )

        response = chat.choices[0].message.content.strip()
        print(f"Received response from OpenAI API for mood metric: {response}")
        
        try:
            result = int(float(response))
            result = max(1, min(5, result))  # Ensure the result is between 1 and 5
            print(f"Converted mood metric to score: {result}")
        except ValueError:
            result = 3  # Default to neutral mood if conversion fails
            print(f"Warning: Couldn't convert '{response}' to an integer. Defaulting to neutral mood (3).")

        return result

def calculate(metric_name, metric_value):
    print(f"Calculating metric: {metric_name}, value: {metric_value}")
    if metric_name == 'cannabis_use':
        agent = CannabisMetricAgent()
        return agent.process_metric(metric_value)
    elif metric_name == 'mood':
        agent = MoodMetricAgent()
        return agent.process_metric(metric_value)
    else:
        # For financial metrics, we'll assume they're already numerical
        try:
            return float(metric_value)
        except ValueError:
            print(f"Warning: Couldn't convert '{metric_value}' to a float for {metric_name}. Returning 0.")
            return 0.0

def get_user_data_as_dataframe(user_id, db):
    print(f"Fetching data for user: {user_id}")
    user_ref = db.collection('users').document(user_id)
    user_doc = user_ref.get()
    
    if not user_doc.exists:
        print(f"No document found for user: {user_id}")
        return None
    
    user_data = user_doc.to_dict()
    if 'state' not in user_data:
        print(f"No 'state' field found in user document for user: {user_id}")
        return None
    
    cannabis_use = user_data['state'].get('cannabis_use_since_last_update', {})
    mood = user_data['state'].get('mood', {})
    monthly_cashflow_income = user_data['state'].get('monthly_cashflow_income', {})
    monthly_cashflow_expenses = user_data['state'].get('monthly_cashflow_expenses', {})
    monthly_cashflow_savings = user_data['state'].get('monthly_cashflow_savings', {})
    monthly_cashflow_savings_rate = user_data['state'].get('monthly_cashflow_savings_rate', {})
    
    print(f"Found {len(cannabis_use)} cannabis use entries, {len(mood)} mood entries, "
          f"{len(monthly_cashflow_income)} income entries, {len(monthly_cashflow_expenses)} expenses entries, "
          f"{len(monthly_cashflow_savings)} savings entries, and {len(monthly_cashflow_savings_rate)} savings rate entries")
    
    # Load existing CSV data from local file
    csv_filename = f"user_{user_id}_data.csv"
    if os.path.exists(csv_filename):
        existing_df = pd.read_csv(csv_filename, parse_dates=['dt'])
        print(f"Loaded existing CSV from local file with {len(existing_df)} rows")
    else:
        existing_df = pd.DataFrame(columns=['dt', 'event_type', 'value'])
        print("Starting with empty DataFrame.")

    df_data = []
    
    for date, use in cannabis_use.items():
        dt = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
        if not ((existing_df['dt'] == dt) & (existing_df['event_type'] == 'cannabis_use_since_last_update_raw')).any():
            print(f"Processing new cannabis use entry for date: {date}")
            processed_value = calculate('cannabis_use', use)
            df_data.append({
                'dt': dt,
                'event_type': 'cannabis_use_since_last_update_raw',
                'value': use
            })
            df_data.append({
                'dt': dt,
                'event_type': 'cannabis_use_since_last_update_grams',
                'value': processed_value
            })
    
    for date, mood_value in mood.items():
        dt = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
        if not ((existing_df['dt'] == dt) & (existing_df['event_type'] == 'mood_raw')).any():
            print(f"Processing new mood entry for date: {date}")
            processed_value = calculate('mood', mood_value)
            df_data.append({
                'dt': dt,
                'event_type': 'mood_raw',
                'value': mood_value
            })
            df_data.append({
                'dt': dt,
                'event_type': 'mood_score',
                'value': processed_value
            })
    
    # Process financial metrics individually
    financial_metrics = [
        ('monthly_cashflow_income', monthly_cashflow_income),
        ('monthly_cashflow_expenses', monthly_cashflow_expenses),
        ('monthly_cashflow_savings', monthly_cashflow_savings),
        ('monthly_cashflow_savings_rate', monthly_cashflow_savings_rate)
    ]

    for metric_name, metric_data in financial_metrics:
        for date, value in metric_data.items():
            dt = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
            if not ((existing_df['dt'] == dt) & (existing_df['event_type'] == metric_name)).any():
                print(f"Processing new {metric_name} entry for date: {date}")
                processed_value = calculate(metric_name, value)
                df_data.append({
                    'dt': dt,
                    'event_type': metric_name,
                    'value': processed_value
                })
    
    new_df = pd.DataFrame(df_data)
    if not new_df.empty:
        new_df = new_df.sort_values('dt')
        print(f"Created DataFrame with {len(new_df)} new rows")
        
        # Combine existing and new data
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.sort_values('dt')
        print(f"Combined DataFrame now has {len(combined_df)} rows")
        
        return combined_df
    else:
        print("No new data to process")
        return existing_df

def update_user_data(user_id):
    db = initialize_firebase()
    print(f"Updating data for user: {user_id}")
    df = get_user_data_as_dataframe(user_id, db)
    if df is None or df.empty:
        print(f"No data found for user {user_id}")
        return None
    
    # Save to CSV locally
    csv_filename = f"user_{user_id}_data.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Data saved to CSV file: {csv_filename}")
    
    return df

def get_quantitative_metrics(df):
    return df[df['event_type'].isin([
        'cannabis_use_since_last_update_grams', 
        'mood_score',
        'monthly_cashflow_income',
        'monthly_cashflow_expenses',
        'monthly_cashflow_savings',
        'monthly_cashflow_savings_rate'
    ])]