import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Cache data for efficient loading
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target
    return df, iris.target_names

# Train RandomForest Classifier
df, target_name = load_data()

# Dataset for model training
X = df.iloc[:, :-1]
y = df["species"]

# Model training for given dataset
model = RandomForestClassifier()
model.fit(X, y)

# Generate test samples from user inputs
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame({'sepal length (cm)': [], 
                                        'sepal width (cm)' : [], 
                                        'petal length (cm)': [], 
                                        'petal length (cm)': [], 
                                        'petal width (cm)' : []})

st.write("Enter values for input features:")
with st.form("add_row_form"):
    sepal_length = st.slider("Sepal length", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
    sepal_width  = st.slider("Sepal width",  float(df['sepal width (cm)'].min()),  float(df['sepal width (cm)'].max()))
    petal_length = st.slider("Petal length", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
    petal_width  = st.slider("Petal width",  float(df['petal width (cm)'].min()),  float(df['petal width (cm)'].max()))

    submitted = st.form_submit_button("Predict")

    if submitted:
        # Prediction            
        input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = model.predict(input_data)
        predicted_species = target_name[prediction[0]]

        new_row = pd.DataFrame([{'sepal length (cm)': sepal_length, 
                                 'sepal width (cm)' : sepal_width, 
                                 'petal length (cm)': petal_length, 
                                 'petal width (cm)': petal_width, 
                                 'predicted species' : predicted_species}])
        
        st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
        
        # Display the prediction
        st.success(f"Row added!   Predicted species is {prediction} {predicted_species}")

st.write("Predicted species for tested samples:")
st.dataframe(st.session_state.df)
