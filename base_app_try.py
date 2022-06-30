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
from turtle import width
import streamlit as st
import streamlit.components.v1 as stc
import joblib, os

#Machine learning dependencies
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

#Data Visualization dependencies
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

#system dependencies
import time
from PIL import Image
import pickle as pkle
import os.path


# Data transformation dependencies
import pandas as pd
import numpy as np
import re


# Load your raw data
raw = pd.read_csv("streamlit_preprocessed.csv", keep_default_na=False)

# creating a sentiment_map and other variables
sentiment_map = {"Anti-Climate": -1, 'Neutral': 0, 'Pro-Climate': 1, 'News-Fact': 2}
type_labels = raw.sentiment.unique()
df = raw.groupby('sentiment')
palette_color = sns.color_palette('dark')

scaler = preprocessing.MinMaxScaler()

#defining function for cleaning raw data
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


# Defining functions for exploratory data analysis
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

    """
    This function helps retrive hashtags, counts it and returns the number of hashtags
    """
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
    '''
    This function analyzes tweets and returns a group of words and frequency based on the selected sentiment
    '''
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
        
    return result


# Defining emoji icons for the main app
pro = Image.open("happy.png")
anti = Image.open("anti.png")
neutral = Image.open("neutral.png")
news_fact = Image.open("news.png")




# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """
    st.set_page_config(page_title="Tweet Classifer", page_icon=":hash:", layout="centered")

    # Setting logo for the app -
    logo = Image.open("Landing.jpg")
    st.image(logo)
    
    # Creating sidebar with selection box -
    menu = ["Landing Page", "Text Prediction", "Upload File", "EDA", "About team"]
    selection = st.sidebar.selectbox("Choose Option", menu)

    # Setting different headers for different page for more interactivity
    if selection == "Landing Page":
        st.markdown('')
    elif selection == "Text Prediction":
        st.subheader("Text Prediction")
    elif selection == "EDA":
        st.subheader("Exploration of Sentiment and Tweets")
    elif selection == "Upload File":
        st.header("Upload File for Prediction")
    else:
        st.subheader("About Team")



    #Creating the Landing page
    landing = Image.open("classify-tweets-1.jpg")
    if selection == "Landing Page":
        st.image(landing)
        time.sleep(3)
        st.subheader("Text Classification App")
                 
        
    #Text Prediction page
    if selection == "Text Prediction":
        st.info("""
        Is your tweet in favour of climate change or not?
        Type or paste them to find out what ECO thinks
        """ )

        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text (Type below)", " ")
        tweet_process = cleaning(tweet_text) # cleaning inputed text

        

        
        if st.button("Classify"): 
            # If classify, makes prediction with model and save output as prediction
            predictor = joblib.load(open(os.path.join("LogisticRegression_model.pkl"), "rb"))
            prediction = predictor.predict([tweet_process])

            # mapping words to prediction. Initially represented as array of number.
            for sen in sentiment_map.keys():
                if sentiment_map.values() == int(prediction):
                    st.success("Text Categorized as: {}".format(sen))

            #defining what will be displayed for each prediction
            if prediction == 2:
                st.write(""" **Kudos!! This shows some news/fact on climate change** 👈 """)
                st.image(news_fact)
            elif prediction == -1:
              st.write(""" **Urgh!! This tweet is most likely is not ECO friendly** 👈 """)
              st.image(anti)
            elif prediction == 1:
                st.write(""" **Yay!!! This tweet is ECO friendly** 👈 """)
                st.image(pro)
            else:
                st.write(""" **Not sure which side are you...Neutral**""")
                st.image(neutral)
                
            
    # Building About Team page
    if selection == "About team":
        st.write("We work with seasoned professionals to give the best product experience")

        st.markdown(" ")
        ken_pic = Image.open("ken.jpg")
        clara_pic = Image.open("Clara-8.jpg")
        emma_pic = Image.open("emmanuel.jpg")


        ken, clara, emma = st.columns(3)

        ken.success("Founder/Growth Strategist")
        clara.success("Product Manager")
        emma.success("Machine Learning Engineer")

        with ken:# ken's profile and picture
            st.header("Kennedy")
            st.image(ken_pic)

            with st.expander("Brief Bio"):
                st.write("""
                Founder of Neural Data Solution. Ken has over 10 years experience as a Business Growth manager possessing additional
                expertis in Product Develpoment. Proficient in facilitating business growth and enhancing market share of 
                the company by leading in-depth market research and competitor analysis, liasing eith senior management and
                conceptualizing new product development. 
                
                Highly skilled in functioning across multiple digital platforms and overseeing
                product design to optimize process. Adept at building businesses and teams from scratch and spearheading Strategy, P&L 
                Management, Marketing and Operations to lead data-driven decision making, render consumer impact analysis and achieve
                astronomical growth with respect to profitability and customer acquisition.
                """)

        with clara: #clara's profile and picture
            st.header("Clara")
            st.image(clara_pic)

            with st.expander("Brief Bio"):
                st.write("""
                Clara is a senior product manager with a background in user experience design and tons of experience in building
                high quality softwares. She has experience with building high quality products and scaling them. Her attention to 
                details is crucial as it has helped to work through models, visualizations, prototypes, requirements and manage across
                functional team. 
                
                She works consistently with Data Scientists, Data Engineers, creatiives and other business-oriented 
                people. She has gathered experience in data analytics, engineering, entrepreneurship, conversion optimization, internet 
                marketing and UX. Using that experience, she has developed a deep understanding of customer journey and product lifecycle.
                """)

        with emma: #Emmanuel's profile and picture
            st.header("Emmanuel")
            st.image(emma_pic)

            with st.expander("Brief Bio"):
                st.write("""
                Emmanuel is a Senor Machine Learning engineer with around 8 years of professional IT experience in Machine Learning
                statistics modelling, Predictive modelling, Data Analytics, Data modelling, Data Architecture, Data Analysis, Data
                mining, Text mining, Natural Language Processing(NLP), Artificial Intelligence algorithms, Business intelligence (BI),
                analytics module (like Decision Trees, Linear and Logistics regression), Hadoop, R, Python, Spark, Scala, MS Excel and SQL.

                He is proficient in managing the entire Data Science project lifecycle and actively involved in the phase of project
                lifecycle including data acquisition, data cleaning, features engineering and statistical modelling.

                """)

        seyi, bodine, moses = st.columns(3)
        bodine.success("Project Manager")
        seyi.success("Lead Software Tester")
        moses.success("Lead Software Developer")

        bodine_pics = Image.open("bodine.jpg")
        seyi_pics = Image.open("seyi.jpg")
        moses_pics = Image.open("moses.jpg")

        with bodine: #Bodine's profile and picture
            st.header("Bodine")
            st.image(bodine_pics)

            with st.expander("Brief Bio"):
                st.write("""
                Bodine is a certified Project Management professional and certified Scrum master with over 5 years experience in 
                project management, project process management, customer service management, marketing and sales. 

                Being a highly motivated and team-oriented professional, she has successfully led large cross-functional team
                to achieve strategic objectives and have managed a team of project managers responsible for implementing
                a project portfolio.
                """)

        with seyi: #Seyi's profile and picture
            st.header("Seyi")
            st.image(seyi_pics)

            with st.expander("Brief Bio"):
                st.write("""
                Seyi is an accomplished Quality Assurance tester with over 3 years experience in Software Testing and Quality Assurance.
                He has a solid understanding in Software Development Life Cycle, Software Testing Lifecycle, bug lifecycle and testing
                diiferent procedure.
                """)

        with moses: #Moses profile and picture
            st.header("Moses")
            st.image(moses_pics)

            with st.expander("Brief Bio"):
                st.write("""
                Moses is a seasoned forwrd looking software engineer with 5+ years backgorund in creating and executing innovative
                software solution to enhance business productivity. Highly experienced in all aspect of the software development 
                lifecycle and end-to-end project management from concept through to development and delivery.

                He is consistently recognized as a hands-on competent leader, skilled at coordinating cross functional team in a 
                fast paced deadline driven environment to steer timely project completion.
                """)



    # Building out the Upload file page
    if selection == "Upload File":
        st.info("Upload a one-columned CSV file that contains group of tweets")

        # declaring options for uploading file
        data_file = st.file_uploader("Upload CSV",type=['csv'])
        if st.button("Process"):
            if data_file is not None:
                df = pd.read_csv(data_file)
                col = df.columns[0]
                tweet_process = df[col].apply(cleaning)
                model_name = 'LogisticRegression_model.pkl'
                predictor = joblib.load(open(os.path.join(model_name), "rb"))
                prediction = predictor.predict(tweet_process)
                pred_data = pd.DataFrame(prediction).value_counts()

                #Writing predictions into a dataframe
                dict_pred = {'Sentiments': ['Pro-climate', 'News', 'Anti', 'Neutral'], 'Predictions': [pred_data[1], pred_data[2], pred_data[-1], pred_data[0]]}
                pd_pred = pd.DataFrame(dict_pred)

                st.write('The breakdown of your analysis is: ')

                pd_pred           
                    
    #Building the EDA page
    if selection == "EDA":
        #creating hash tag segment
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

            fig, ax = plt.subplots(1,2, figsize=(12, 5)) # plotiing frequency bar graph
            ax[0].bar(data=source, height=result.values(), x= result.keys(), color = '#00b5dd')
            ax[0].set_xticklabels(result.keys(), rotation=75)
                       
            #building wordcloud
            word_cloud = WordCloud(#background_color='white',
                                   width=900,
                                   height=900).generate(' '.join(result.keys()))
            ax[1].imshow(word_cloud)
            ax[1].axis("off")
            plt.show()
            st.pyplot(fig, use_container_width=True)

        # creating word groupings    
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
