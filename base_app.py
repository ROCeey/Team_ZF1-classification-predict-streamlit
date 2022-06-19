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
import joblib, os
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Data dependencies
import pandas as pd
import numpy as np

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl", "rb")
tweet_cv = joblib.load(news_vectorizer)  # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("streamlit_preprocessed.csv", keep_default_na=False)

# creating a sentiment_map and other variables
sentiment_map = {"Anti-Climate": -1, 'Neutral': 0, 'Pro-Climate': 1, 'News-Fact': 2}
type_labels = raw.sentiment.unique()
df = raw.groupby('sentiment')


def bag_of_words_count(words, word_dict={}):
    """ this function takes in a list of words and returns a dictionary
        with each word as a key, and the value represents the number of
        times that word appeared"""
    words = words.split()
    for word in words:
        if word in word_dict.keys():
            word_dict[word] += 1
        else:
            word_dict[word] = 1
    return word_dict


def hash_tag(sentiment_cat=1, iter_hash_num=5, labels=type_labels, dataframe=df):
    sentiment_dict = {}
    counter = 0
    for pp in labels:
        sentiment_dict[pp] = {}
        for row in dataframe.get_group(pp)['hash_tag']:
            sentiment_dict[pp] = bag_of_words_count(row, sentiment_dict[pp])
    result = {}
    for w in sorted(sentiment_dict[sentiment_cat], key=sentiment_dict[sentiment_cat].get, reverse=True):
        counter += 1
        result[w] = sentiment_dict[sentiment_cat][w]
        # print(w, sentiment_dict[sentiment_cat][w])
        if counter >= iter_hash_num:
            break
    return result


def word_grouping(group_word_num=3, sentiment_cat=1, ngram_iter_num=3, dataframe=df):
    ngram_dict = {}
    # converting each word in the dataset into features
    vectorized = CountVectorizer(analyzer="word", ngram_range=(group_word_num, group_word_num),
                                 max_features=1000)  # setting the maximum feature to 8000
    reviews_vect = vectorized.fit_transform(dataframe.get_group(sentiment_cat)['cleaned_tweet'])
    features = reviews_vect.toarray()
    # Knowing the features that are present
    vocab = vectorized.get_feature_names_out()
    # Sum up the counts of each vocabulary word
    dist = np.sum(features, axis=0)

    # For each, print the vocabulary word and the number of times it
    for tag, count in zip(vocab, dist):
        ngram_dict[tag] = count
    # Creating an iteration
    most_pop = iter(sorted(ngram_dict, key=ngram_dict.get, reverse=True))
    result = {}
    for x in range(ngram_iter_num):
        most_pop_iter = next(most_pop)
        result[most_pop_iter] = ngram_dict[most_pop_iter]
        # print(most_pop_iter, ngram_dict[most_pop_iter])
    return result


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
	#st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information", "About team"]
	selection = st.sidebar.selectbox("Choose Option", options)

	if selection == "Prediction":
		st.subheader('Prediction')
	elif selection == "Information":
		st.subheader("Information")
	else:
		st.subheader("About Team")

	#Building About Team page
	if selection == "About Team":
		col5, col6, col7= st.columns(3)


		#bodine, seyi, moses = st.columns(3)

	# with st.container():
    # 	st.write("This is inside the container")
		
	

	# Building out the "Information" page
	if selection == "Information":
		st.info("Brief Description")

		#st.slider("select a range of numbers", 0, 10)
		# You can read a markdown file from supporting resources folder
		#st.markdown("Some information here")

		st.markdown(" ")

		#st.container()

		col1, col2 = st.columns(2)
		col1.success('1')
		col2.success('Important/most used words')

		with col1:
			st.slider("select a range of numbers", 0, 10)
			st.checkbox('sentiment 1', value = False)#, key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False)
			st.checkbox('sentiment 2', value = False)#, key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False)
			st.checkbox('sentiment 3', value = False)#, key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False)
			st.checkbox('sentiment 4', value = False)#, key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False)

		with col2:
			st.write('The word cloud function goes in here')

		st.markdown(" ")
			

		col3, col4 = st.columns(2)
		col3.success('Popular hashtags')
		col4.success('mentions')

		with col3:
			st.write("List of popular hashtags function associated with sentiment goes in here")

		with col4:
			chart_data = pd.DataFrame(np.random.randn(50, 3), columns=["a", "b", "c"])
			
			st.bar_chart(chart_data)

		# with st.container():
		
		# 	
				

		
		
			#st.write("Important words/most used words")

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
