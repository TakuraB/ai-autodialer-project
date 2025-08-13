import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- 1. Load the Data ---
print("Loading data...")
try:
    df = pd.read_csv('data/twcs.csv', dtype={'in_response_to_tweet_id': str, 'tweet_id': str})
    print(f"1. Initial rows loaded: {len(df)}")
except FileNotFoundError:
    print("ERROR: 'data/twcs.csv' not found. Make sure the file is in the 'data' directory.")
    exit()

# --- 2. Separate Customer and Agent Tweets ---
# Inbound tweets are from customers
customer_tweets = df[df['inbound'] == True].copy()
# Outbound tweets are from support agents
agent_tweets = df[df['inbound'] == False].copy()

print(f"2. Separated into {len(customer_tweets)} customer tweets and {len(agent_tweets)} agent tweets.")

# --- 3. Match Conversations ---
# We want to find pairs where an agent's tweet is a reply to a customer's tweet.
# We do this by merging where the agent's 'in_response_to_tweet_id' matches the customer's 'tweet_id'.
merged_df = pd.merge(
    agent_tweets,
    customer_tweets,
    left_on='in_response_to_tweet_id',
    right_on='tweet_id',
    suffixes=('_agent', '_customer')
)
print(f"3. Found {len(merged_df)} matched conversation pairs.")

# --- 4. Clean and Finalize the Dataset ---
# Select and rename the columns we need
df_final = merged_df[['text_customer', 'text_agent']].rename(columns={
    'text_customer': 'customer_message',
    'text_agent': 'agent_response'
})

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("\nCleaning text...")
df_final['customer_message'] = df_final['customer_message'].apply(clean_text)
df_final['agent_response'] = df_final['agent_response'].apply(clean_text)

# Drop any rows that became empty after cleaning
df_final.dropna(inplace=True)
df_final = df_final[(df_final['customer_message'] != '') & (df_final['agent_response'] != '')]
print(f"4. Final rows after cleaning and removing empty messages: {len(df_final)}")

# --- 5. Save the Final Dataset ---
output_path = 'cleaned_customer_support_data.csv'
df_final.to_csv(output_path, index=False)

print("\nCleaned Data Example:")
print(df_final.head())
print(f"\nSuccessfully saved {len(df_final)} rows to '{output_path}'")
