import streamlit as st
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
import joblib
from transformers import TFBertModel

emotion_columns_list = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Set up tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load trained model using joblib
@st.cache(allow_output_mutation=True)
def load_trained_model():
    # Custom objects needed for Keras model deserialization
    custom_objects = {'TFBertModel': TFBertModel}
    
    # Load the components using joblib
    model_architecture = joblib.load('model_architecture.joblib')
    model_weights = joblib.load('model_weights.joblib')
    optimizer_state = joblib.load('optimizer_state.joblib')

    # Reconstruct the model architecture with custom objects
    model = tf.keras.models.model_from_json(model_architecture, custom_objects=custom_objects)
    
    # Load the model weights and optimizer state
    model.set_weights(model_weights)
    optimizer = tf.keras.optimizers.Adam.from_config(optimizer_state)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Tokenize and pad the input text
def tokenize_single_text(text):
    tokens = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        return_tensors='tf'
    )
    return {
        'input_ids': np.array(tokens['input_ids']),
        'attention_mask': np.array(tokens['attention_mask'])
    }

def main():
    st.title("Emotion Recognition")

    # Introduction and GitHub link
    st.markdown(
        "Welcome to the BERT-based Emotion Classifier App!\n\n"
        "This app uses a BERT-based model to classify emotions in text.\n\n"
        "The model is trained on Google's GoEmotions dataset, a fine-grained emotion dataset. "
        "You can learn more about the dataset in [Google's Blog](https://ai.googleblog.com/2021/10/goemotions-dataset-for-fine-grained.html).\n\n"
        "Feel free to explore the code on [GitHub](https://github.com/Savvy-Slowley/GoEmotions)."
    )
    
    user_input = st.text_area("Type the text you'd like to classify:")

    if len(user_input) > 125:
        st.warning("Input text is too long. Please limit it to 125 characters.")
        return

    if st.button('Predict Emotion'):
        # Load the trained model
        model = load_trained_model()

        tokens = tokenize_single_text(user_input)
        predictions = model.predict(tokens)
        predicted_labels = (predictions > 0.5).astype(int)[0]

        predicted_emotions = [emotion for emotion, label in zip(emotion_columns_list, predicted_labels) if label]

        st.write("Predicted Emotions:", ', '.join(predicted_emotions))


if __name__ == '__main__':
    main()
