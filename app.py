import streamlit as st

# Load and serve the image
with open("https://github.com/sarvesh-pro/AMDProjects/blob/main/AMD2.jpg", "rb") as f:
    st.image(f.read(), use_column_width=True)

import streamlit as st
import numpy as np
import joblib
import base64


# Function to get the base64 string of an image
def get_base64_image(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


# Load your pre-trained model (make sure 'linear_regression_model.pkl' exists)
model = joblib.load('linear_regression_model.pkl')

# Path to the local image
img_path = "AMDProjects/AMD2.jpg"
img_base64 = get_base64_image(img_path)

# Title of the web app
st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url("data:image/jpeg;base64,{img_base64}") no-repeat center center fixed;
        background-size: cover;
    }}
    .main {{
        padding: 5rem;
    }}
    .title {{
        color: #FFFFFF;
        font-size: 2rem;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }}
    .input-container {{
        background-color: rgba(0, 0, 0, 0.5);
        padding: 2rem;
        border-radius: 10px;
        margin: auto;
        width: fit-content;
    }}
    .input-container input, .input-container button {{
        margin-bottom: 1rem;
    }}
    .prediction {{
        color: #FFD700;
        font-size: 1.5rem;
        text-align: center;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Title of the web app
st.markdown('<div class="title">AMD Users Per Session Prediction</div>', unsafe_allow_html=True)

# Container for inputs and prediction
with st.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)

    # Input fields for your features
    feature_1 = st.number_input('Enter Situation Room Level:', format="%.f")
    feature_2 = st.number_input('Enter Type of Device:', format="%.f")

    # When the user clicks "Predict"
    if st.button('Predict'):
        # Reshape input to the model's expected format
        input_data = np.array([[feature_1, feature_2]])

        # Make a prediction
        prediction = model.predict(input_data)

        # Convert prediction to a float for formatting
        predicted_value = float(prediction[0])

        # Display the prediction
        st.markdown(f'<div class="prediction">The predicted users per session are : {predicted_value:.2f}</div>',
                    unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
