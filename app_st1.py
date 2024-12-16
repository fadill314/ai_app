import streamlit as st

st.sidebar.image("http://www.ehtp.ac.ma/images/lo.png")
st.sidebar.header("Executive Master Cloud Computing")

st.header("App pour test des services Azure AI")
st.subheader("Application 1 : Image Analysis")


import os
from PIL import Image, ImageDraw
import sys
#from matplotlib import pyplot as plt
from azure.core.exceptions import HttpResponseError
import requests

# Import namespaces
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

# Get Config

ai_endpoint = st.secrets['AI_SERVICE_ENDPOINT']
ai_key = st.secrets['AI_SERVICE_KEY']

# Get image
image_file = st.file_uploader('Charger une image',type=['png', 'jpg'])
if image_file is not None:
    image_data = image_file.getvalue()
    
    # Authenticate Azure AI Vision client
    cv_client = ImageAnalysisClient(
        endpoint=ai_endpoint,
        credential=AzureKeyCredential(ai_key)
        )
        
       
    # Background removal
    #BackgroundForeground(ai_endpoint, ai_key, image_file)

     
    def AnalyzeImage(image_filename, image_data, cv_client):
        st.write('\nAnalyzing image...')

        try:
            # Get result with specified features to be retrieved
            result = cv_client.analyze(
            image_data=image_data,
            visual_features=[
            VisualFeatures.CAPTION,
            VisualFeatures.DENSE_CAPTIONS,
            VisualFeatures.TAGS,
            VisualFeatures.OBJECTS,
            VisualFeatures.PEOPLE],
        )

        except HttpResponseError as e:
            st.write(f"Status code: {e.status_code}")
            st.write(f"Reason: {e.reason}")
            st.write(f"Message: {e.error.message}")

        # Display analysis results
        # Get image captions
        if result.caption is not None:
            st.write("\nCaption:")
            st.write(" Caption: '{}' (confidence: {:.2f}%)".format(result.caption.text, result.caption.confidence * 100))

    # Get image dense captions
        if result.dense_captions is not None:
            st.write("\nDense Captions:")
            for caption in result.dense_captions.list:
                st.write(" Caption: '{}' (confidence: {:.2f}%)".format(caption.text, caption.confidence * 100))

    # Analyze image
    AnalyzeImage(image_file, image_data, cv_client)

    st.sidebar.title("Sidebar Title")
    st.sidebar.markdown("This is the sidebar content")
