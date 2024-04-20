import streamlit as st
import os
import numpy as np
import pandas as pd
from PIL import Image
import joblib

## Title
st.title("Iris Flower Dataset")

## Sidebar --- Today's Iris Prediction
# with st.sidebar.expander("Today's Iris Flower prediction"):
st.header("Today's Iris Flower Prediction")

pred_col, actual_col = st.columns(2)

with pred_col:
    pred_image = Image.open('pred_assets/latest_iris.jpg')
    st.image(pred_image, caption="Prediction from model")

with actual_col:
    actual_image = Image.open('pred_assets/actual_iris.jpg')
    st.image(actual_image, caption="Actual image")

st.write("Data Source: Synthetic Data")
st.write("Update Frequency: Daily")

# with st.sidebar.expander("Performance monitoring"):
st.header("Performance monitoring")

st.subheader("Recent prediction/outcomes")
recpo = Image.open('pred_assets/df_recent.png')
st.image(recpo, caption="Recent prediction/Outcomes")

st.subheader("Confusion matrix of historical model")
cm = Image.open("pred_assets/confusion_matrix.png")
st.image(cm, caption="Confusion matrix of historical model")


with st.sidebar.expander("Iris Demo"):
    st.header("Iris demo")
    st.write("Insert details of iris flower below and get predictions")

    # Unpickle the model
    iris_model = joblib.load("iris_model/iris_model.pkl")

    # Enter input
    petal_length = st.number_input(label="Petal Length (cm): ", min_value=0.0, max_value=20.0, value=1.0, step=0.01)
    petal_width = st.number_input(label="Petal Width (cm): ", min_value=0.0, max_value=20.0, value=1.0, step=0.01)
    sepal_length = st.number_input(label="Sepal Length (cm): ", min_value=0.0, max_value=20.0, value=1.0, step=0.01)
    sepal_width = st.number_input(label="Sepal Width (cm): ", min_value=0.0, max_value=20.0, value=1.0, step=0.01)

    # make prediction function
    def make_prediction(sepal_length, sepal_width, petal_length, petal_width):
        sample = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, 4)
        
        df = pd.DataFrame(
            sample, 
            columns=[
                'sepal_length', 'sepal_width', 
                'petal_length', 'petal_width'])
    
        pred = iris_model.predict(df)
        
        pred = ['setosa', 'versicolor', 'virginica'][pred[0]]
    
        predd = f"Iris {pred}"
    
        # Load and return image
        image_path = os.path.join('images', f'{pred}.jpg')
        image = Image.open(image_path)    
    
        return predd, image

    if st.button("Predict"):
        prediction, image = make_prediction(sepal_length, sepal_width, petal_length, petal_width)
        st.success(f"The flower is an {prediction}")
        st.image(image, caption=f"{prediction} flower")