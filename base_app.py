"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies and processing functions
import pandas as pd
import numpy as np # required for processing functions
import string
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer

# Vectorizer
# news_vectorizer = open("resources/tfidfvect.pkl","rb")
# tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Vectorizer modified
news_vectorizer = open("resources/c_vectorizer_fin.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# Processing cleaning functions
# Replace web-urls with more common string.
def replace_urls(df):
    pattern_url = r"http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+"
    subs_url = r"url-web"
    df["message"] = df["message"].replace(to_replace = pattern_url, value = subs_url, regex = True)
    return df

# Convert all string entries to lower case.
def to_lower(df):
    df["message"] = df["message"].str.lower()
    return df

# Cleans dataframe column to contain only valid characters.
normal = "abcdefghijklmnopqrstuvwxyz " + """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
def keep_normal_chars(df):
    df["message"] = df["message"].apply(lambda message: "".join([l for l in message if l in normal]))
    return df

# Remove punctuation as per string.punctuation.
def remove_all_punctuation(df):
	df["message"] = df["message"].apply(lambda message: "".join([l for l in message if l not in string.punctuation]))
	return df

# Removes short words.
def min_length(df, l=1):
    df["message"] = df["message"].apply(lambda x: x.split())
    df["message"] = df["message"].apply(lambda x: " ".join([w for w in x if len(w) > l]))
    return df

# Tokenise string in dataframe.
def tree_tokenise(df):
    tokeniser = TreebankWordTokenizer()
    df["message"] = df["message"].apply(tokeniser.tokenize)
    return df

# Convert word tokens to lemmatized words.
def df_lemma(df):
    lemmatizer = WordNetLemmatizer()
    df["message"] = df["message"].apply(lambda row: [lemmatizer.lemmatize(w_in_row) for w_in_row in row])
    return df

# Join tokenised words in list to single string.
def join_tokens(df): 
    df["message"] = df["message"].apply(lambda row: " ".join(row))
    return df


# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Model - Logistic Regression and Count Vectorizer.")
		st.info("Cleaning functions used are:-  \n replace urls" \
			"  \n to lower case" \
			"  \n keep normal characters - no numbers" \
			"  \n remove all punctuation" \
			"  \n min length of word to keep" \
			"  \n tree tokeniser" \
			"  \n lemmatise tokens" \
			"  \n join all tokens")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Clean the input with processing functions
			tt_df = pd.DataFrame(np.array([tweet_text]), columns=["message"])
			st.write(tt_df[['message']])
			clean_tt_df = replace_urls(tt_df)
			clean_tt_df = to_lower(clean_tt_df)
			clean_tt_df = keep_normal_chars(clean_tt_df)
			clean_tt_df = remove_all_punctuation(clean_tt_df)
			clean_tt_df = min_length(clean_tt_df, l=1)
			clean_tt_df = tree_tokenise(clean_tt_df)
			clean_tt_df = df_lemma(clean_tt_df)
			clean_tt_df = join_tokens(clean_tt_df)
			
			st.write(clean_tt_df[['message']])
			vect_text = tweet_cv.transform(clean_tt_df["message"]).toarray()

			# Transforming user input with vectorizer (original)
			# vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			# predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			# prediction = predictor.predict(vect_text)

			# Model modified
			predictor = joblib.load(open(os.path.join("resources/model_log_refr_fin.pkl"),"rb")) # load trained model
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
