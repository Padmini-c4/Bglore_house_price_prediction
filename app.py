#Streamlit UI

#import neceessary libraries

import pickle
import streamlit as st 
import pandas as pd 

#Model De-serialization (loading model)

with open("Linear_model.pkl","rb") as file:
    model = pickle.load(file)

#model.predict(data)


#import joblib
#file = "model.pkl"
#model=joblib.load(file)
#model.predict(data)

#encoded de-serialization(loading encoder)
with open("label_encoded.pkl","rb") as file1:
    encoder = pickle.load(file1)
    

#load cleaned dataset

df = pd.read_csv("cleaned_data.csv")


st.set_page_config(page_title="house price prediction bangalore",page_icon="https://thumbs.dreamstime.com/b/green-houses-community-model-abstract-real-estate-logo-vector-professional-architecture-company-design-115154734.jpg")
with st.sidebar:
    st.title("Bengalore House Price Prediction")
    st.image("data:image/jpeg;base64,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")
    

#input fields

#'location','bhk','total_sqft','bath','encoded_loc'
location = st.selectbox("Location: ",options=df["location"].unique())

BHK = st.selectbox("BHK: ",options=sorted(df["bhk"].unique()))

total_sqft = st.number_input("Total_sqft: ",min_value=300)

bath = st.selectbox("No. of Restrooms: ",options=sorted(df["bath"].unique()))

#encoded the new location
encoded_loc = encoder.transform([location])

#new data preparation
new_data = [[BHK,total_sqft,bath,encoded_loc[0]]]

#prediction
col1,col2 = st.columns([1,2])

if col2.button("Predict House Price"):
    pred = model.predict(new_data)[0]
    pred = round(pred*100000)
    st.subheader(f"Predicted Price : Rs. {pred}")