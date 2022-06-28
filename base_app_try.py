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
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import base64
import time
from PIL import Image
import pickle as pkle
import os.path


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
thumps_down = Image.open("thums_down.webp")
thumps_up = Image.open("thumps_up.webp")
nuetral = Image.open("neutral.webp")
news_fact = Image.open("news.webp")

file_.close()


# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """
    st.set_page_config(page_title="Tweet Classifer", page_icon=":hash:", layout="centered")

    # Creates a main title and subheader on your page -
    logo = Image.open("Landing.jpg")
    st.image(logo)
    #st.title("Eco")
    # st.subheader("Climate change tweet classification")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    menu = ["Landing Page", "Text Prediction", "Upload File", "EDA", "Company Profile", "About team"]
    selection = st.sidebar.selectbox("Choose Option", menu)


    if selection == "Landing Page":
        st.markdown('')
    elif selection == "Text Prediction":
        st.subheader("Text Prediction")
    elif selection == "EDA":
        st.subheader("Exploration of Sentiment and Tweets")
    elif selection == "Upload File":
        st.header("Upload File for Prediction")
    elif selection == "Company Profile":
        st.header("Company Profile")
    else:
        st.subheader("About Team")



    #Landing page
    landing = Image.open("classify-tweets-1.jpg")
    if selection == "Landing Page":
        st.image(landing)#, height=1500)
        time.sleep(3)
        st.subheader("Text Classification App")
        st.button("Go to next page")

            
        
    #Text Prediction page
    if selection == "Text Prediction":
        st.info("Prediction with ML Models")

        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text (Type below)", " ")
        
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

        
        if st.button("Classify"):
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
            predictor = joblib.load(open(os.path.join(model_map[model_name]), "rb"))
            prediction = predictor.predict([tweet_process])

            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            for sen in sentiment_map.keys():
                if sentiment_map.values() == int(prediction):
                    st.success("Text Categorized as: {}".format(sen))

            # if prediction == 1:
            #     st.write(""" **Thank you for supporting climate** 👈 """)
            #     st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
            #                 unsafe_allow_html=True)
            if prediction == 2:
                st.write(""" **This is a news fact about climate change** 👈 """)
                st.image(news_fact)
            elif prediction == -1:
              st.write(""" **This tweet does not believe in climate change** 👈 """)
              st.image(thumps_down)
            elif prediction == 1:
                st.write(""" **This tweet supports climate change** 👈 """)
                st.image(thumps_up)
            else:
                st.write(""" **Neutral**""")
                st.image(nuetral)
    # # # st.markdown('---')
    # model_col,Accuracy_col=st.columns(2)
    # Accuracy_col.header('**Model Matrics**')

    # Accuracy_col.subheader('mean absolute error')
    # Accuracy_col.write(mean_absolute_error(y_test,prediction))
    # Accuracy_col.subheader('mean square error')
    # Accuracy_col.write(mean_squared_error(y_test,prediction))
    # Accuracy_col.subheader('R squared score error')
    # Accuracy_col.write(r2_score(y,prediction))

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

        with ken:
            st.header("Kennedy")
            st.image(ken_pic)

            with st.expander("Brief Bio"):
                st.write("""
                Founder of TechNation.Inc. Ken has over 10 years experience as a Business Growth manager possessing additional
                expertis in Product Develpoment. Proficient in facilitating business growth and enhancing market share of 
                the company by leading in-depth market research and competitor analysis, liasing eith senior management and
                conceptualizing new product development. 
                
                Highly skilled in functioning across multiple digital platforms and overseeing
                product design to optimize process. Adept at building businesses and teams from scratch and spearheading Strategy, P&L 
                Management, Marketing and Operations to lead data-driven decision making, render consumer impact analysis and achieve
                astronomical growth with respect to profitability and customer acquisition.
                """)

        with clara:
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

        with emma:
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

        with bodine:
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

        with seyi:
            st.header("Seyi")
            st.image(seyi_pics)

            with st.expander("Brief Bio"):
                st.write("""
                Seyi is an accomplished Quality Assurance tester with over 3 years experience in Software Testing and Quality Assurance.
                He has a solid understanding in Software Development Life Cycle, Software Testing Lifecycle, bug lifecycle and testing
                diiferent procedure.
                """)

        with moses:
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



# Required to let Streamlit instantiate our web app.

    # Building out the "Information" page
    # if selection == "Information":
    #     st.info("Brief Description")

    #     st.markdown(" ")

    #     # hash_pick = st.checkbox('Hash-Tag')
    #     # if hash_pick:
    #     #     val = st.selectbox("Choose Tag type", ['Hash-Tag', 'Mentions'])
    #         # sentiment_select = st.selectbox("Choose Option", sentiment_map)
    #         # iter_hash_select = st.slider('How many hash-tag', 1, 20, 10)

    #         # if val == 'Hash-Tag':
    #         #     st.info("Popular Hast Tags")
    #         # else:
    #         #     st.info("Popular Mentions")
    #         # valc = 'hash_tag' if val == 'Hash-Tag' else 'mentions'
    #         # result = tags(sentiment_cat=sentiment_map[sentiment_select], iter_hash_num=iter_hash_select,
    #         #               col_type=valc)
        

        
    #     # result = tags(sentiment_cat=sentiment_map[sentiment_select], iter_hash_num=iter_hash_select,
    #     #                   col_type=valc)

    #     col1, col2 = st.columns(2)
    #     col1.success('1')
    #     col2.success('Important/most used words')

    #     with col1:
    #         sentiment_select = st.selectbox("Choose Option", sentiment_map)
    #         iter_hash_select = st.slider('How many hash-tag', 1, 20, 10)
            
           
    #     with col2:
    #         st.write('The word cloud function goes in here')

    #         st.markdown(" ")


    #     col3, col4 = st.columns(2)
    #     col3.success('Popular hashtags')
    #     col4.success('mentions')

    #     with col3:
    #         source = pd.DataFrame({
    #             'Frequency': result.values(),
    #             'Hash-Tag': result.keys()})
    #         st.write("List of popular hashtags function associated with sentiment goes in here")

    #     with col4:
    #         chart_data = pd.DataFrame(np.random.randn(50, 3), columns=["a", "b", "c"])

    #         st.bar_chart(chart_data)



    # Building out the prediction page
    if selection == "Upload File":
        st.info("Prediction with ML Models")

        data_file = st.file_uploader("Upload CSV",type=['csv'])
        if st.button("Process"):
            if data_file is not None:
                df = pd.read_csv(data_file)
                tweet_process = df['message'].apply(cleaning)
                # vectorizer = CountVectorizer(analyzer = "word", max_features = 8000)
                # reviews_vect = vectorizer.fit_transform(df['cleaned_tweet'])
                model_name = 'naive_bayes_model.pkl'
                predictor = joblib.load(open(os.path.join(model_name), "rb"))
                prediction = predictor.predict(tweet_process)
                
                
                st.success(
                    pd.DataFrame(prediction).value_counts().plot(kind='bar'))
                plt.show()
                #st.pyplot(fig, use_container_width=True) 
                



        # Creating a text box for user input
        # st.markdown('---')
        # tweet_text = st.text_area("Enter Text (Type in the box below)", " ")
        # st.markdown('---')
        # model_name = st.selectbox("Choose Model", model_map.keys())
        # tweet_process = cleaning(tweet_text)

        # st.write('You selected:', model_name)

        # if model_name == 'LogisticRegression':
        #     with st.expander("See explanation"):
        #         st.write("""Brief explanation of how the Logistic regression works goes in here""")

        # elif model_name == 'KNeighborsClassifier':
        #     with st.expander("See explanation"):
        #         st.write("""Brief explanation of how the KNN model works goes in here""")

        # elif model_name == 'SVC':
        #     with st.expander("See explanation"):
        #         st.write("""Brief explanation of how the SVC model works goes in here""")

        # elif model_name == 'DecisionTreeClassifier':
        #     with st.expander("See explanation"):
        #         st.write("""Brief explanation of how the Decision tree classifier model works goes in here""")

        # else:
        #     with st.expander("See explanation"):
        #         st.write("""Brief explanation of how the model works goes in here""")

        # st.markdown('---')

    #     if st.button("Classify"):
    #         # Load your .pkl file with the model of your choice + make predictions
    #         # Try loading in multiple models to give the user a choice
    #         predictor = joblib.load(open(os.path.join(model_map[model_name]), "rb"))
    #         prediction = predictor.predict([tweet_process])

    #         # When model has successfully run, will print prediction
    #         # You can use a dictionary or similar structure to make this output
    #         # more human interpretable.
    #         st.success("Text Categorized as: {}".format(prediction))
    #         if prediction == 1:
    #             st.write(""" **Thank you for supporting climate** 👈 """)
    #             st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
    #                         unsafe_allow_html=True)
    #     # # # st.markdown('---')
    #     # model_col,Accuracy_col=st.columns(2)
    #     # Accuracy_col.header('**Model Matrics**')

    # # Accuracy_col.subheader('mean absolute error')
    # # Accuracy_col.write(mean_absolute_error(y_test,prediction))
    # # Accuracy_col.subheader('mean square error')
    # # Accuracy_col.write(mean_squared_error(y_test,prediction))
    # Accuracy_col.subheader('R squared score error')
    # Accuracy_col.write(r2_score(y,prediction))

    if selection == "EDA":
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
            fig, ax = plt.subplots(1,2, figsize=(10, 15))
            ax[0].pie(data=source, x=result.values(), labels=result.keys(), colors=palette_color)
                       #explode=dd[0], autopct='%.0f%%')
            word_cloud = WordCloud(#background_color='white',
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
