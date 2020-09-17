import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from PIL import Image

# image = Image.open('sparks\iris_flowers.png')
# st.image(image, caption='Sunrise by the mountains',
#        use_column_width=True)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# loading the trained model
filename = os.path.join(BASE_DIR,'sparks\sk_tree.sav')
model = pickle.load(open(filename, 'rb'))

home_page = 1

st.sidebar.title('Decision Tree')

if st.sidebar.button('Decision Tree Demo'):
    home_page = 1

if st.sidebar.button('Tree Visualization'):
    home_page = 0
    st.markdown('# Visualizing the trained Decision Tree')
    st.image('sparks\sk-tree.png', use_column_width=True)

if st.sidebar.button('Algorithm overview'):
    home_page = 0
    st.markdown('# Decision Tree algorithm overview')

    st.write("The tree basically works by asking questions to partition the dataset.The data points satisfying the question are separated as true or false nodes. Then Depending on how consistent or inconsistent the values at particular node are, it may be further partioned till the values are distributed to the appropriate classes")

    st.markdown('### The two main calculations which decide the data partition are:')
    st.markdown('<ul><li>Gini Impurity</li> <li>Information Gain</li></ul>',unsafe_allow_html=True)

    st.markdown('### Gini Impurity')
    st.write('It calculates the impurity or entropy at a particular node of the tree')
    st.image('sparks\gini.png', use_column_width=True)
    # displaying code on the webpage
    with st.echo():
        def gini(rows):
            counts = class_counts(rows)
            impurity = 1
            for lbl in counts:
                prob_of_lbl = counts[lbl] / float(len(rows))
                impurity -= prob_of_lbl ** 2
            return impurity

    st.markdown('### Information Gain')
    st.write('Calculating the information gain will decide which question will yield less impurity and arrive at the best question to be chosen as root node.')
    st.write('#### The uncertainty of the starting node, minus the weighted impurity of two child nodes')
    st.image('sparks\info_gain.png', use_column_width=True)
    with st.echo():
        def info_gain(left, right, current_uncertainty):
            p = float(len(left)) / (len(left) + len(right))
            return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

    st.markdown('### Using the above calculations the tree is built recursively by finding the best split')
    st.write('Finding the best question to ask by iterating over every feature / value and calculating the best gain.')
    with st.echo():
        def find_best_split(rows):

            best_gain = 0  # keep track of the best information gain
            best_question = None  # keep train of the feature / value that produced it
            current_uncertainty = gini(rows)
            n_features = len(rows[0]) - 1  # number of columns

            for col in range(n_features):  # for each feature

                values = set([row[col] for row in rows])  # unique values in the column

                for val in values:  # for each value

                    question = Question(col, val)

                    # try splitting the dataset
                    true_rows, false_rows = partition(rows, question)

                    # Skip this split if it doesn't divide the
                    # dataset.
                    if len(true_rows) == 0 or len(false_rows) == 0:
                        continue

                    # Calculate the information gain from this split
                    gain = info_gain(true_rows, false_rows, current_uncertainty)

                    if gain >= best_gain:
                        best_gain, best_question = gain, question

            return best_gain, best_question

    st.markdown('<h4>To view the complete Decision Tree algorithm implemented from scratch visit my github:</h4> <a href="https://github.com/prajvalsudhir/Sparks-Streamlit/blob/master/Task_4_Decison_tree.ipynb">Decision Tree</a> ',unsafe_allow_html=True)










if home_page:
    st.markdown("<center><h1>Decision Tree Classifier</h1></center>", unsafe_allow_html=True)
    st.image('sparks\iris_flowers.png', use_column_width=True)

    st.markdown('## Enter the values below to get a prediction from the classifier')

    st.markdown('### Sepal length')
    sepal_length = st.slider('', min_value=4.3, max_value=7.9, value=None, step=None, format=None, key=None)
    print('sepal_length:', sepal_length)

    st.markdown('### Sepal width')
    sepal_width = st.slider('', min_value=2.0, max_value=4.4, value=None, step=None, format=None, key=None)
    print('sepal_width:', sepal_width)

    st.markdown('### Petal length')
    petal_length = st.slider('', min_value=1.0, max_value=6.9, value=None, step=None, format=None, key=None)
    print('petal_length:', petal_length)

    st.markdown('### Petal width')
    petal_width = st.slider('', min_value=0.1, max_value=2.5, value=None, step=None, format=None, key=None)
    print('petal_width:', petal_width)

    if st.button('PREDICT'):
        st.markdown('### Result:')
        result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        print(result[0].upper())
        st.markdown(f'## {result[0].upper()}', unsafe_allow_html=True)


