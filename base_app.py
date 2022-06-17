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
import base64
# Data dependencies
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# """### gif from local file"""
file_ = open("thank_you.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()



# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """
	st.set_page_config(page_title="Tweet Classifer", page_icon=":hash:", layout="centered")

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.markdown('---')
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")
	st.markdown('---')

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
		col1, col2 = st.columns(2)
		
		col1.subheader("Widgets will go here")
		age =col1.slider('Number of words?', 0, 100, 25)
		tweet_news =col1.radio('Select three known variables:',['1 Pro',' -1 Anti','0 Neutral','2 News'])

		col2.subheader("word Cloud visualization")
	

	# Building out the predication page
	if selection == "Prediction":

		st.info("Prediction with ML Models")
		# Creating a text box for user input
		st.markdown('---')
		tweet_text = st.text_area("Enter Text","Type Here")
		st.markdown('---')
	
		option = st.selectbox('Please select your model?',(
			'LogisticRegression','KNeighborsClassifier',
		    'SVC',
		    'DecisionTreeClassifier',
		    'RandomForestClassifier',
		    'AdaBoostClassifier',
		    'MLPClassifier',
		    'LinearSVC'))
		st.write('You selected:', option)
		st.markdown('---')

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))
			if prediction==1:
				st.write(""" **Thank you for supporting climate** 👈 """)
				st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
				unsafe_allow_html=True)
		# st.markdown('---')
		# model_col,Accuracy_col=st.columns(2)
		# Accuracy_col.header('**Model Matrics**')
		
		# Accuracy_col.subheader('mean absolute error')
		# Accuracy_col.write(mean_absolute_error(y_test,prediction))
		# Accuracy_col.subheader('mean square error')
		# Accuracy_col.write(mean_squared_error(y_test,prediction))
		# Accuracy_col.subheader('R squared score error')
		# Accuracy_col.write(r2_score(y,prediction))


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
