import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
from sklearn.datasets import fetch_california_housing

st.title('🏠House Price prediction using ML')
st.sidebar.image('https://i.pinimg.com/originals/93/c7/44/93c744bcde1780c94bb1d3f03991f8a7.gif')

df = pd.read_csv('house_data.csv')
X = df.iloc[:,:-3]
y = df.iloc[:,-1]

st.sidebar.title('Select House Features')
st.image('https://i.pinimg.com/originals/93/c7/44/93c744bcde1780c94bb1d3f03991f8a7.gif')

all_value = []
for i in X:
 min_value = int(X[i].min())
 max_value = int(X[i].max())
 
 ans = st.sidebar.slider(f'Select {i} value', min_value, max_value)
 all_value.append(ans)

# st.write(all_value)

