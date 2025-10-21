import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🧫 Bacteria Colony Counter")

uploaded_file = st.file_uploader("Upload an image of the Petri dish", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image)

    st.image(img, caption="Uploaded Petri Dish", use_column_width=True)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # ✅ Adjusted adaptive threshold
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 2
    )

    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ✅ More lenient filter
    filtered_contours = [c for c in contours if 5 < cv2.contourArea(c) < 5000]

    colony_count = len(filtered_contours)

    result = img.copy()
    cv2.drawContours(result, filtered_contours, -1, (255,0,0), 2)

    st.image(result, caption=f"Detected Colonies: {colony_count}", use_column_width=True)
    st.success(f"✅ Estimated number of colonies: {colony_count}")
    

