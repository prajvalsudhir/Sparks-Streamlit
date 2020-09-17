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

st.markdown("<center><h1>Decision Tree Classifier</h1></center>",unsafe_allow_html=True)
st.image('sparks\iris_flowers.png',use_column_width=True)

st.markdown('## Enter the values below to get a prediction from the classifier')

st.markdown('### Sepal length')
sepal_length = st.slider('',min_value=4.3, max_value=7.9, value=None, step=None, format=None, key=None)
print('sepal_length:',sepal_length)

st.markdown('### Sepal width')
sepal_width = st.slider('',min_value=2.0, max_value=4.4, value=None, step=None, format=None, key=None)
print('sepal_width:',sepal_width)

st.markdown('### Petal length')
petal_length = st.slider('',min_value=1.0, max_value=6.9, value=None, step=None, format=None, key=None)
print('petal_length:',petal_length)

st.markdown('### Petal width')
petal_width = st.slider('',min_value=0.1, max_value=2.5, value=None, step=None, format=None, key=None)
print('petal_width:',petal_width)


if st.button('PREDICT'):
    st.markdown('### Result:')
    result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    print(result[0].upper())
    st.markdown(f'## {result[0].upper()}',unsafe_allow_html=True)


# if st.sidebar.button('Decsision Tree Image'):
#     st.markdown('# Tree')