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
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import base64

# Data dependencies
import pandas as pd
import numpy as np
import re

# Model_map
model_map = {'KNeighborsClassifier': 'KNeighborsC_model.pkl', 'Naive_bayes': 'naive_bayes_model.pkl'}

# Load your raw data
raw = pd.read_csv("streamlit_preprocessed.csv", keep_default_na=False)

# creating a sentiment_map and other variables
sentiment_map = {"Anti-Climate": -1, 'Neutral': 0, 'Pro-Climate': 1, 'News-Fact': 2}
type_labels = raw.sentiment.unique()
df = raw.groupby('sentiment')
palette_color = sns.color_palette('dark')

scaler = preprocessing.MinMaxScaler()


def cleaning(tweet):
    """The function uses patterns with regular expression, 'stopwords'
        from natural language processing (nltk) and  tokenize using split method
        to filter and clean each tweet message in a dataset"""

    pattern = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
    rem_link = re.sub(pattern, '', tweet)
    rem_punct = re.sub(r'[^a-zA-Z ]', '', rem_link)
    rem_punct = re.sub(r'RT', '', rem_punct)
    word_split = rem_punct.lower().split()
    stops = set(stopwords.words("english"))
    without_stop_sent = ' '.join([t for t in word_split if t not in stops])
    return without_stop_sent


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


def tags(sentiment_cat=1, iter_hash_num=5, labels=type_labels, dataframe=df, col_type: str = 'hash_tag'):
    sentiment_dict = {}
    counter = 0
    for pp in labels:
        sentiment_dict[pp] = {}
        for row in dataframe.get_group(pp)[col_type]:
            sentiment_dict[pp] = bag_of_words_count(row, sentiment_dict[pp])
    result = {}
    for w in sorted(sentiment_dict[sentiment_cat], key=sentiment_dict[sentiment_cat].get, reverse=True):
        counter += 1
        result[w] = sentiment_dict[sentiment_cat][w]
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
    # st.subheader("Climate change tweet classification")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Prediction", "Information", "EDA", "About team"]
    selection = st.sidebar.selectbox("Choose Option", options)

    if selection == "Prediction":
        st.subheader('Prediction')
    elif selection == "Information":
        st.subheader("Information")
    elif selection == "EDA":
        st.subheader('Exploratory Data Analysis')
    else:
        st.subheader("About Team")

    # Building About Team page
    if selection == "About team":
        st.write("Meet our amazing team")

        st.markdown(" ")

        ken, clara, emma = st.columns(3)

        ken.success("Role")
        clara.success("Role")
        emma.success("Role")

        with ken:
            st.header("Meet Kennedy")
            st.image("https://static.streamlit.io/examples/cat.jpg")

            with st.expander("Brief Bio"):
                st.write("""Bio goes in here""")

        with clara:
            st.header("Meet Clara")
            st.image("https://static.streamlit.io/examples/dog.jpg")

            with st.expander("Brief Bio"):
                st.write("""Bio goes in here""")

        with emma:
            st.header("Meet Emmanuel")
            st.image("https://static.streamlit.io/examples/owl.jpg")

            with st.expander("Brief Bio"):
                st.write("""Bio goes in here""")

        bodine, seyi, moses = st.columns(3)
        bodine.success("Role")
        seyi.success("Role")
        moses.success("Role")

        with bodine:
            st.header("Meet Bodine")
            st.image("https://static.streamlit.io/examples/cat.jpg")

            with st.expander("Brief Bio"):
                st.write("""Bio goes in here""")

        with seyi:
            st.header("Meet Seyi")
            st.image("https://static.streamlit.io/examples/dog.jpg")

            with st.expander("Brief Bio"):
                st.write("""Bio goes in here""")

        with moses:
            st.header("Meet Moses")
            st.image("https://static.streamlit.io/examples/owl.jpg")

            with st.expander("Brief Bio"):
                st.write("""Bio goes in here""")

    # with st.container():
    # 	st.write("This is inside the container")

    # Building out the "Information" page
    if selection == "Information":
        st.info("Brief Description")

        # st.slider("select a range of numbers", 0, 10)
        # You can read a markdown file from supporting resources folder
        # st.markdown("Some information here")

        st.markdown(" ")

        # st.container()

        col1, col2 = st.columns(2)
        col1.success('1')
        col2.success('Important/most used words')

        with col1:
            st.slider("select a range of numbers", 0, 10)
            st.checkbox('sentiment 1',
                        value=False)  # , key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False)
            st.checkbox('sentiment 2',
                        value=False)  # , key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False)
            st.checkbox('sentiment 3',
                        value=False)  # , key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False)
            st.checkbox('sentiment 4',
                        value=False)  # , key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False)

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

        # st.write("Important words/most used words")

        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'):  # data is hidden if box is unchecked
            st.write(raw[['sentiment', 'message']])  # will write the df to the page
        col1, col2 = st.columns(2)

        col1.subheader("Widgets will go here")
        age = col1.slider('Number of words?', 0, 100, 25)
        tweet_news = col1.radio('Select three known variables:', ['1 Pro', ' -1 Anti', '0 Neutral', '2 News'])

        col2.subheader("word Cloud visualization")

    # Building out the predication page
    if selection == "Prediction":

        st.info("Prediction with ML Models")
        # Creating a text box for user input
        st.markdown('---')
        tweet_text = st.text_area("Enter Text", "Type Here")
        st.markdown('---')
        model_name = st.selectbox("Choose Model", model_map.keys())
        tweet_process = cleaning(tweet_text)

        st.write('You selected:', model_name)

        if model_name == 'LogisticRegression':
            with st.expander("See explanation"):
                st.write("""Brief explanation of how the Logistic regression works goes in here""")

        elif model_name == 'KNeighborsClassifier':
            with st.expander("See explanation"):
                st.write("""Brief explanation of how the KNN model works goes in here""")

        elif model_name == 'SVC':
            with st.expander("See explanation"):
                st.write("""Brief explanation of how the SVC model works goes in here""")

        elif model_name == 'DecisionTreeClassifier':
            with st.expander("See explanation"):
                st.write("""Brief explanation of how the Decision tree classifier model works goes in here""")

        else:
            with st.expander("See explanation"):
                st.write("""Brief explanation of how the model works goes in here""")

        st.markdown('---')

        if st.button("Classify"):
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
            predictor = joblib.load(open(os.path.join(model_map[model_name]), "rb"))
            prediction = predictor.predict([tweet_process])

            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("Text Categorized as: {}".format(prediction))
            if prediction == 1:
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

    if selection == "EDA":
        st.subheader("Exploration of Sentiment and Tweets")
        hash_pick = st.checkbox('Hash-Tag')
        if hash_pick:
            val = st.selectbox("Choose Tag type", ['Hash-Tag', 'Mentions'])
            sentiment_select = st.selectbox("Choose Option", sentiment_map)
            iter_hash_select = st.slider('How many hash-tag', 1, 20, 10)
            if val == 'Hash-Tag':
                st.info("Popular Hast Tags")
            else:
                st.info("Popular Mentions")
            valc = 'hash_tag' if val == 'Hash-Tag' else 'mentions'
            result = tags(sentiment_cat=sentiment_map[sentiment_select], iter_hash_num=iter_hash_select,
                          col_type=valc)
            source = pd.DataFrame({
                'Frequency': result.values(),
                'Hash-Tag': result.keys()
            })
            val = np.array(list(result.values())).reshape(-1, 1)
            dd = (scaler.fit_transform(val)).reshape(1, -1)
            fig, ax = plt.subplots(2, figsize=(10, 15))
            ax[0].pie(data=source, x=result.values(), labels=result.keys(), colors=palette_color,
                      explode=dd[0], autopct='%.0f%%')
            word_cloud = WordCloud(background_color='white',
                                   width=512,
                                   height=384).generate(' '.join(result.keys()))
            ax[1].imshow(word_cloud)
            ax[1].axis("off")
            plt.show()
            st.pyplot(fig, use_container_width=True)

        word_pick = st.checkbox('Word Group(s)')
        if word_pick:
            st.info("Popular Group of Word(s)")
            sentiment_select_word = st.selectbox("Choose sentiment option", sentiment_map)
            word_amt = st.slider('Group of words', 1, 10, 5)
            group_amt = st.slider("Most frequent word groupings", 1, 10, 5)
            word_result = word_grouping(group_word_num=word_amt, ngram_iter_num=group_amt,
                                        sentiment_cat=sentiment_map[sentiment_select_word])
            st.table(pd.DataFrame({
                'Word group': word_result.keys(),
                'Frequency': word_result.values()
            }))


# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
