import streamlit as st
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def caption_generator(image):
    """
    input : image_file
    output: text
    """
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0],skip_special_tokens=True)
    return caption

st.title("画像解説AI")
uploaded_file = st.file_uploader("画像ファイルを選んでください")

if uploaded_file:
    with st.spinner("画像を処理中..."):
        try:
            img = Image.open(uploaded_file)
            caption = caption_generator(img)
            st.image(img, caption='アップロードされた画像') 
            st.write(caption)

        except Exception as e:
            st.write("ファイルを再度アップロードしてください")
            st.write(f"エラー詳細: {e}")



