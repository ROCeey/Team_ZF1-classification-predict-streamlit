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

Vectorizer
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


# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """

    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("Tweet Classifier")
    # st.subheader("Climate change tweet classification")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Prediction", "Information", "EDA"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Information":
        st.subheader("Climate change tweet classification")
        st.info("General Information")
        # You can read a markdown file from supporting resources folder
        st.markdown("Some information here")

        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'):  # data is hidden if box is unchecked
            st.write(raw[['sentiment', 'message']])  # will write the df to the page

    # Building out the predication page
    if selection == "Prediction":
        st.subheader("Climate change tweet classification")
        st.info("Prediction with ML Models")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text", "Type Here")

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
            predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"), "rb"))
            prediction = predictor.predict(vect_text)

            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("Text Categorized as: {}".format(prediction))

    if selection == "EDA":
        st.subheader("Exploration of Sentiment and Tweets")
        hash_pick = st.checkbox('Hash-Tag')
        if hash_pick:
            st.info("Popular Hast Tags")
            # labels = st.selectbox("Choose Option", type_labels)
            sentiment_select = st.selectbox("Choose Option", sentiment_map)
            iter_hash_select = st.slider('How many hash-tag', 1, 30, 10)
            result = hash_tag(sentiment_cat=sentiment_map[sentiment_select], iter_hash_num=iter_hash_select)
            source = pd.DataFrame({
                'Frequency': result.values(),
                'Hash-Tag': result.keys()
            })
            fig, ax = plt.subplots(2, figsize=(10, 15))
            sns.barplot(data=source, y='Frequency', x='Hash-Tag', ax=ax[0])
            wordcloud = WordCloud().generate(' '.join(result.keys()))
            ax[1].imshow(wordcloud)
            xlabels = source['Hash-Tag']
            ax[0].tick_params(axis='x', labelrotation=75)
            ax[1].axis("off")
            plt.show()
            st.pyplot(fig, use_container_width=True)
        word_pick = st.checkbox('Word Group(s)')

        if word_pick:
            st.info("Popular Group of Word(s)")
            sentiment_select_word = st.selectbox("Choose sentiment option", sentiment_map)
            word_amt = st.slider('Number of words', 1, 10, 5)
            group_amt = st.slider("Numbers of word groupings", 1, 10, 5)
            word_result = word_grouping(group_word_num=word_amt, ngram_iter_num=group_amt,
                                        sentiment_cat=sentiment_map[sentiment_select_word])
            st.json(word_result)


# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
