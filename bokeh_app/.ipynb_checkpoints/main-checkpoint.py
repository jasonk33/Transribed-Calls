import pandas as pd
from joblib import load
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import LayoutDOM, Button
from bokeh.models.widgets import Div
from bokeh.core.properties import String

# Code to allow you to select an input file
file_input = """
import * as p from "core/properties"
import {LayoutDOM, LayoutDOMView} from "models/layouts/layout_dom"

export class FileInputView extends LayoutDOMView
  initialize: (options) ->
    super(options)
    input = document.createElement("input")
    input.type = "file"
    input.onchange = () =>
      @model.value = input.value
    @el.appendChild(input)

export class FileInput extends LayoutDOM
  default_view: FileInputView
  type: "FileInput"
  @define {
    value: [ p.String ]
  }
"""

# For file input
class FileInput(LayoutDOM):
    __implementation__ = file_input
    value = String()
input = FileInput()

# Initial text to be displayed
prediction_text = Div(text="<b>Prediction: </b>", style={'font-size': '300%'}, width=1000)
probability_text = Div(text="<b>Probability: </b>", style={'font-size': '200%'}, width=1000)

# Load model
model = load("model.joblib")

# Function to upload the data after the user clicks the upload button
def upload_data():

    # Load the text file
	with open("text_files/{}".format(input.value[12:]), 'r') as f:
		text = f.read()
		
	# Initialize dataframe object to format data for model prediction
	df = pd.DataFrame()
		
	# Binary features for specific words occuring in the transcribed call
	words_in_text = ["create", "before", "again", "new", "first", "past", "last", "never", "phone", "number", "cell", "address", "email", "credit", 
					 "between", "week", "today", "@"]
	
	# Create feature for each word
	for word_in_text in words_in_text:
		df["{}_in_call".format(word_in_text)] = [int(word_in_text in text)]
	
	# Get number of words in each call
	df['num_words_in_call'] = [len(text.split())]
	
	# Features
	numeric_features = ['num_words_in_call']
	categorical_features = ['create_in_call', 'before_in_call', 'again_in_call', 'new_in_call', 'first_in_call', 'past_in_call', 'last_in_call', 
							'never_in_call', 'phone_in_call', 'number_in_call', 'cell_in_call', 'address_in_call', 'email_in_call', 'credit_in_call', 
							'between_in_call', 'week_in_call', 'today_in_call', '@_in_call']
	
	# Get data in final format
	X = df[numeric_features + categorical_features]

	# Make prediction using model
	prediction_probability = model.predict_proba(X)[0,1]
	
	# Display text based on model prediction
	if prediction_probability > .5:
		prediction_text.text = "<b>Prediction: New Customer</b>"
	else:
		prediction_text.text = "<b>Prediction: Returning Customer</b>"
	probability_text.text = "<b>Probability: {}</b>".format(max(prediction_probability, 1-prediction_probability))

# Button to upload text file
upload_botton = Button(label="Upload")

# Function to be called when upload button is clicked
upload_botton.on_click(upload_data)

# Add all elements to page
curdoc().add_root(column(prediction_text, probability_text, input, upload_botton))
     
def main():
    pass

if __name__ == '__main__':
    main()
