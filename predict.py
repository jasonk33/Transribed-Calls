# Import necessary packages
import pandas as pd
import numpy as np
from joblib import load
import json
import warnings
warnings.filterwarnings("ignore")


# Read in data
df = pd.read_csv("TDL_FullDataSet.csv")

# Drop rows with no tags
df = df.dropna(subset=['Tags'])

# Get output variable
df['New_Customer'] = df.Tags.str.contains("New Customer").astype(int)

# Get number of words in each call
df['num_words_in_call'] = df.text.str.split().apply(len)

# Convert date to usable format
df['Date.Time'] = pd.to_datetime(df['Date.Time'])

# Get date features
df['Hour'] = df['Date.Time'].dt.hour
df['Day_Of_Week'] = df['Date.Time'].dt.dayofweek
df['Month'] = df['Date.Time'].dt.month

# Binary features for specific words occuring in the transcribed call
words_in_text = ["profile", "create", "before", "again", "new", "first", "repeat", "file", "past", "last", "never", "ever", 
                 "phone", "number", "cell", "address", "home", "email", "account", "payment", "credit", "information", "ago", 
                 "between", "week", "month", "year", "today", "tomorrow", "yesterday", "@", "com", '10', 'available', 'bad', 
                "name", "set", "looked", 'refer', 'hold', 'contact', 'regarding', 'located', 'zip', 'time', 'price', 'offer',
                'provide']

# Create feature for each word
for word_in_text in words_in_text:
    df["{}_in_call".format(word_in_text)] = df.text.str.contains(word_in_text).astype(int)

# Get the duration of the call in seconds	
def get_duration(row):
    time = row['Duration'].split(":")
    tot_seconds = int(time[2])
    tot_seconds += int(time[0])*3600
    tot_seconds += int(time[1])*60
    return tot_seconds
df['duration_seconds'] = df.apply(get_duration, axis=1)

numeric_features = ['duration_seconds', 'num_words_in_call']
categorical_features = ['create_in_call', 'before_in_call', 'again_in_call', 'new_in_call', 'first_in_call', 'past_in_call', 'last_in_call', 
                        'never_in_call', 'phone_in_call', 'number_in_call', 'cell_in_call', 'address_in_call', 'email_in_call', 'credit_in_call', 
                        'between_in_call', 'week_in_call', 'today_in_call', '@_in_call']
X = df[numeric_features + categorical_features]

model = load("model.joblib")

preds = model.predict_proba(X)[:,1]

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
with open('preds.json', 'w') as f:
    json.dump(preds, f, cls=NumpyEncoder)