# Import necessary packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from joblib import dump
import warnings
warnings.filterwarnings("ignore")


def preprocess_data(df, numeric_features, categorical_features, ordinal_features, ordinal_categories, label, random_state=7):
	"""Helper function to preprocess data. Builds pipeline step for feature scaling, one hot encoding, and split of train/test and validation sets.
        ----------
        df : pandas dataframe object
        numeric_features : list of numerical feature names
		categorical_features : list of categorical feature names
		numeric_features : list of numerical feature names
		ordinal_features : list of ordinal feature names
		ordinal_categories : list of ordinal feature categories (list of lists)
        label : name of the label column
        random_state : int, optional
            random seed to use
        Returns
        -------
        X_cv : X for cross validation 
		X_validation : X for validation
		y_cv : y for cross validation
		y_validation : y for validation
		preprocessor : pipeline step for preprocessing
    """
	
	# Combine all input features
	X = df[numeric_features + categorical_features]
	
	# Encode the output label
	le = LabelEncoder()
	y = le.fit_transform(df[label])
    
    # Create the preprocessing pipelines for both numeric and categorical data
	preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features),
            ('ord', OrdinalEncoder(categories = ordinal_categories), ordinal_features)])
    
	# Create the train/test and validation sets
	X_cv, X_validation, y_cv, y_validation = train_test_split(X, y, test_size=0.2, random_state=random_state**1, stratify=y)

	# Return data and pipeline step
	return X_cv, X_validation, y_cv, y_validation, preprocessor


def cross_validate_model(model, param_grid, preprocessor, X_cv, y_cv, n_splits=5, n_repeats=1, random_state=7, preprocess=True, verbose=5):
	"""Helper function to perform repeated k fold cross validation for a model. Uses a grid search to find optimal hyperparameters.
        ----------
        model : a sklearn model to fit
		param_grid : a dictionary of hyperparameter values
		preprocessor : pipeline step for preprocessing
		X_cv : X for cross validation 
		y_cv : y for cross validation
		n_splits : int, optional
            k to use for k fold cross validation
		n_repeats : int, optional
            number of times to repeat cross validation
		random_state : int, optional
            k to use for k fold cross validation
		preprocess : int, boolean
            perform preprocessing steps
		verbose : int, optional
            amount of printing
        Returns
        -------
        grid : all results from cross validation
    """
    
	# Get all hyperparameters
	keys = list(param_grid.keys())
	for key in keys:
		param_grid['classifier__' + key] = param_grid.pop(key)
    
	# Generate pipeline
	if preprocess:
		pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
	else:
		pipe = Pipeline(steps=[('classifier', model)])
        
	# Perform grid search for hyperparameters using repeated k fold cross validation
	grid = GridSearchCV(pipe, param_grid, cv=RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state**2), n_jobs=1, return_train_score=True, 
                                scoring=['accuracy', 'neg_log_loss'], refit='accuracy', verbose=verbose)

	# Fit to data
	grid.fit(X_cv, y_cv)
    
	# Return all results of model training
	return grid

def get_cross_validation_results(grid, X_validation, y_validation):
	"""Helper function to analyze cross validation results and print out metrics.
        ----------
        grid : all results from cross validation
		X_validation : X for validation
		y_validation : y for validation
    """
	
	# Get index of best result
	best_param_idx = grid.cv_results_['rank_test_accuracy'].argmin()
	
	# Get metrics
	cv_log_loss = -grid.cv_results_['mean_test_neg_log_loss'][best_param_idx]
	cv_accuracy = grid.best_score_
	
	# Get best hyperparameter combination
	cv_params = grid.best_params_
	keys = list(cv_params.keys())
	for key in keys:
		cv_params[key.split("__")[1]] = cv_params.pop(key)
	validation_accuracy = grid.score(X_validation, y_validation)
    
	# Print out all info
	print("Cross Validation Results:\n\tAccuracy: {}\n\tLog Loss: {}\n\tBest Parameters: {}\nValidation Accuracy: {}".format(
        cv_accuracy, cv_log_loss, cv_params, validation_accuracy))


### Feature engineering ###

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

# Feature names
numeric_features = ['duration_seconds', 'num_words_in_call']
categorical_features = ['create_in_call', 'before_in_call', 'again_in_call', 'new_in_call', 'first_in_call', 'past_in_call', 'last_in_call', 
                        'never_in_call', 'phone_in_call', 'number_in_call', 'cell_in_call', 'address_in_call', 'email_in_call', 'credit_in_call', 
                        'between_in_call', 'week_in_call', 'today_in_call', '@_in_call']
ordinal_features = []
ordinal_categories = [[]]

# Label name
label = 'New_Customer'


### Machine Learning Pipeline ###

# Preprocess data
X_cv, X_validation, y_cv, y_validation, preprocessor =  preprocess_data(df, numeric_features, categorical_features, ordinal_features, ordinal_categories, label)

# Baseline accuracy
classes, counts = np.unique(y_validation,return_counts=True)
print('balance:',np.max(counts/len(y_validation)))

# Train Different Models #

# Logistic Regression (simple linear model)
model = LogisticRegression(random_state=123)
print(type(model).__name__)
# Hyperparameters to try
param_grid = {'C': [.01, .05, .1, .5, 1]}
# Perform model training, selecting best hyperparameters through cross validation
grid_LR = cross_validate_model(model, param_grid, preprocessor, X_cv, y_cv, n_splits=5, n_repeats=1, verbose=0)
# Print results
get_cross_validation_results(grid_LR, X_validation, y_validation)

# Random Forest
model = RandomForestClassifier(random_state=123)
print(type(model).__name__)
# Hyperparameters to try
param_grid = {'n_estimators': [10, 50, 100], 'max_depth': [5,7,9]}
# Perform model training, selecting best hyperparameters through cross validation
grid_RF = cross_validate_model(model, param_grid, preprocessor, X_cv, y_cv, n_splits=5, n_repeats=1, verbose=0)
# Print results
get_cross_validation_results(grid_RF, X_validation, y_validation)

# K Nearest Neighbors
model = KNeighborsClassifier()
print(type(model).__name__)
# Hyperparameters to try
param_grid = {'n_neighbors': [10, 25, 50]}
# Perform model training, selecting best hyperparameters through cross validation
grid_KNN = cross_validate_model(model, param_grid, preprocessor, X_cv, y_cv, n_splits=5, n_repeats=1, verbose=0)
# Print results
get_cross_validation_results(grid_KNN, X_validation, y_validation)

# Support Vector Machine (this one takes a long time, uncomment if want to use)
# model = SVC(random_state=123, probability=True)
# print(type(model).__name__)
# param_grid = {'C': [25, 50, 75], 'gamma': [1e-3, 1e-2, 1e-1]}
# grid_SVC = cross_validate_model(model, param_grid, preprocessor, X_cv, y_cv, n_splits=5, n_repeats=1, verbose=10)
# get_cross_validation_results(grid_SVC, X_validation, y_validation)

# Gradient Boosting
model = GradientBoostingClassifier(random_state=123)
print(type(model).__name__)
# Hyperparameters to try
param_grid = {'max_depth': [3,5,7], 'n_estimators': [50,100,150], 'learning_rate': [.05,.1,.2]}
# Perform model training, selecting best hyperparameters through cross validation
grid_GBC = cross_validate_model(model, param_grid, preprocessor, X_cv, y_cv, n_splits=5, n_repeats=1, verbose=0)
# Print results
get_cross_validation_results(grid_GBC, X_validation, y_validation)

# Take best model and save it
dump(grid_GBC.best_estimator_, "model.joblib")