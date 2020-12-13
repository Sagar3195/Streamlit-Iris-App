import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

def user_input():
  sepal_length = st.sidebar.slider("Sepal Length", 4.3,7.9,5.0)  #min value, max value, initial value
  sepal_width = st.sidebar.slider("Sepal Width", 2.0,4.4,3.0)  #min value, max value, initial value
  petal_length = st.sidebar.slider("Petal Length", 1.0,6.9,3.0)  #min value, max value, initial value
  petal_width = st.sidebar.slider("Petal Width",  0.1,2.5,1.0)  #min value, max value, initial value
  data = {"sepal_length": sepal_length,
          "sepal_width": sepal_width,
          "petal_length": petal_length,
          "petal_width": petal_width}
  
  features = pd.DataFrame(data, index = [0])
  return features 
 
st.write("# Simple Iris Flower Prediction App")
st.write("# Irisi Dataset")
st.write("Number of Classes: 3")
st.write("Classifier : KNN")
st.sidebar.header("User Input Parameters")
st.subheader("User Input Parameters")

df = user_input()
st.write(df)

iris = datasets.load_iris()
x = iris.data
y = iris.target

##Apply the algorithm
model =  KNeighborsClassifier(n_neighbors = 5)
##Train the model
model.fit(x,y)
##predict the model
predictor = model.predict(df)

st.subheader("Class labels and their corresponding index number: ")
st.write(iris.target_names)


st.subheader("Prediction")
st.write(iris.target_names[predictor])
