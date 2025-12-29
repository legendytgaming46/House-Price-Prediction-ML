import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
st.title('🏠 House Price prediction using ML')
st.image('https://i.pinimg.com/originals/d3/18/13/d3181322e4522cf897fa8c1a038c6a2d.gif')
df = pd.read_csv('house_data.csv')
X = df.iloc[:,:-3]
y = df.iloc[:,-1]

st.sidebar.title('🏠 Select House features')
st.sidebar.image('https://i.pinimg.com/originals/bd/a3/ed/bda3edbcdde12a14c477648671e6b9d2.gif')
all_value = []
for i in X:
  min_value = int(X[i].min())
  max_value = int(X[i].max())
  ans = st.sidebar.slider(f'Select {i} value',min_value, max_value)
  all_value.append(ans)

# st.write(all_value)

scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)
final_value = scaler.transform([all_value])

@st.cache_data
def model_run(X,y):
  model = RandomForestRegressor()
  model.fit(X,y)
  return model

model = model_run(X,y)  
house_price = model.predict(final_value)[0]

with st.spinner('Predicting House price'):
  time.sleep(1)
msg = f'''House price is: $ {round(house_price*100000,2)}'''
st.success(msg)

st.markdown('''**Design and Developed by: Aditya Gupta**''')



