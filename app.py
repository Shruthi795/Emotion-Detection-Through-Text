import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import os

st.set_page_config(page_title="Text Emotion Detection", page_icon=":smiley:", layout="centered", 
                   initial_sidebar_state="collapsed")

model_path = r"C:/Users/laxmi/Sravani/project1/EmotionDetection.pkl"

st.markdown(
    """
    <style>
    .header-container {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        height: 50px;
    }
    /* Hide the "Deploy" button in the top-right */
    header > div:nth-child(2) > div {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

pipe_lr = None

if os.path.exists(model_path):
    try:
        pipe_lr = joblib.load(model_path)
    except Exception as e:
        st.write(f"Error loading model from {model_path}: {e}")
else:
    st.write(f"Model file not found at {model_path}")

# Emoji dictionary for emotions
emotions_emoji_dict = {
    "anger": "😠",
    "fear": "😨😱",
    "joy": "😂",
    "sadness": "😔",
    "surprise": "😮",
    "love":"❤️"
}

# Function to predict emotions
def predict_emotions(docx):
    if pipe_lr is not None:
        try:
            results = pipe_lr.predict([docx])
            return results[0]
        except Exception as e:
            st.write(f"Error in prediction: {e}")
            return None
    else:
        st.write("Model is not loaded properly.")
        return None

# Function to get prediction probabilities
def get_prediction_proba(docx):
    if pipe_lr is not None:
        try:
            results = pipe_lr.predict_proba([docx])
            return results
        except Exception as e:
            st.write(f"Error in prediction probability: {e}")
            return None
    else:
        st.write("Model is not loaded properly.")
        return None

# Main function
def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotions In Text")

    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here", label_visibility='collapsed')
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        if prediction:
            probability = get_prediction_proba(raw_text)

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write(f"{prediction}: {emoji_icon}")
                st.write(f"Confidence: {np.max(probability)}")

            with col2:
                st.success("Prediction Probability")
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(
                    x='emotions',
                    y='probability',
                    color='emotions'
                )
                st.altair_chart(fig, use_container_width=True)
        else:
            st.error("Failed to make a prediction. Please check the model and input text.")

if __name__ == '__main__':
    main()
