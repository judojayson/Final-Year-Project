import streamlit as st

st.set_page_config(page_title="Alzheimer’s Info Chatbot", page_icon="💬", layout="centered")

st.title("💬 Alzheimer’s Assistant Chatbot")
st.markdown("Ask anything about Alzheimer’s, gait, symptoms, or our model!")

# Predefined Q&A pairs
qa = {
    "what is alzheimer's": "Alzheimer’s is a progressive neurodegenerative disease that leads to memory loss, confusion, difficulty with language, and impaired judgment. It primarily affects adults over 65.",
    "at what age does alzheimer's occur": "Most people with Alzheimer’s are 65 or older, but early-onset forms can appear as early as in a person’s 40s or 50s.",
    "what are the symptoms of alzheimer's": "Common symptoms include memory loss, confusion, difficulty performing familiar tasks, changes in mood or personality, and trouble understanding visual images.",
    "is there a cure for alzheimer's": "There is no cure for Alzheimer’s yet. However, medications like donepezil or memantine can help manage symptoms, and lifestyle changes can slow progression.",
    "what are risk factors for alzheimer's": "Major risk factors include age, genetics (especially the APOE-e4 gene), head injury, lack of exercise, poor diet, and social isolation.",
    "what is gait analysis": "Gait analysis is the study of human walking patterns. In neurology, it can help detect abnormalities that signal conditions like cerebellar disorders or Alzheimer’s.",
    "what gait styles indicate neuro problems": "Cerebellar gait, which involves wide steps and poor balance, is often associated with neurological issues such as Alzheimer’s and Parkinson’s.",
    "how does your project predict alzheimer's": "Our project uses two approaches:\n\n- Vision Transformer & CNN to classify MRI brain scans.\n- MediaPipe & Transformers to analyze gait from video.\n\nEach model predicts independently and their performance is compared.",
    "how many gait landmarks are used": "We extract 66 features from 33 body landmarks (x and y coordinates) per frame using MediaPipe pose estimation.",
    "what is grad-cam": "Grad-CAM (Gradient-weighted Class Activation Mapping) visualizes which areas of an image contribute most to a model's prediction — used here for MRI model interpretability.",
    "how accurate is your model": "Our ViT model achieved ~94.5% accuracy, CNN ~91.2%. For gait, Transformer reached 92.5%, and Logistic Regression ~96% accuracy.",
    "how do i use this app": "Upload an MRI scan or a gait video, or both. The system will display predictions, confidence scores, model comparison charts, and heatmaps if applicable.",
}

# Handle chat
user_input = st.text_input("Type your question here:")

if user_input:
    response = "I'm sorry, I didn't understand that. Please try a simpler question."
    for question, answer in qa.items():
        if question in user_input.lower():
            response = answer
            break
    st.markdown(f"**🤖 Answer:** {response}")
