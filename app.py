import streamlit as st
import pandas as pd
import joblib
## from vision import estimate_height

# Load models
stunting_model = joblib.load("stunting_model.pkl")
wasting_model = joblib.load("wasting_model.pkl")

# Load encoders
gender_encoder = joblib.load("gender_encoder.pkl")
stunting_encoder = joblib.load("stunting_encoder.pkl")
wasting_encoder = joblib.load("wasting_encoder.pkl")

st.title("NutriVision")

st.info(
    """
    Instructions:

    • Stand upright

    • Hold a white A4 paper vertically

    • Show the entire body

    • Wear clothing that contrasts with the paper

    • Ensure both feet are visible
    """
)

uploaded_file = st.file_uploader(
    "Upload Child Image",
    type=["jpg", "jpeg", "png"]
)

height = None

if uploaded_file is not None:

    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    if uploaded_file is not None:
        from vision import estimate_height

        height = estimate_height("temp.jpg")

    if height is not None:
        st.success(f"Estimated Height: {height:.1f} cm")

gender = st.selectbox(
    "Gender",
    gender_encoder.classes_
)

age = st.number_input(
    "Age (months)",
    min_value=0,
    max_value=60,
    value=24
)

weight = st.number_input(
    "Weight (kg)",
    min_value=1.0,
    max_value=50.0,
    value=10.0
)
if st.button("Analyze") and height is not None:

    gender_encoded = gender_encoder.transform([gender])[0]

    input_data = pd.DataFrame(
        [[
            gender_encoded,
            age,
            height,
            weight
        ]],
        columns=[
            "Jenis Kelamin",
            "Umur (bulan)",
            "Tinggi Badan (cm)",
            "Berat Badan (kg)"
        ]
    )

    stunting_pred = stunting_model.predict(input_data)[0]
    wasting_pred = wasting_model.predict(input_data)[0]

    stunting_probs = stunting_model.predict_proba(input_data)[0]
    wasting_probs = wasting_model.predict_proba(input_data)[0]

    stunting_confidence = max(stunting_probs) * 100
    wasting_confidence = max(wasting_probs) * 100

    stunting_result = stunting_encoder.inverse_transform(
        [stunting_pred]
    )[0]

    wasting_result = wasting_encoder.inverse_transform(
        [wasting_pred]
    )[0]

    st.success("Assessment Complete")

    st.write(
        f"**Stunting Status:** {stunting_result}"
    )

    st.write(
        f"Confidence: {stunting_confidence:.2f}%"
    )

    st.write(
        f"**Wasting Status:** {wasting_result}"
    )

    st.write(
        f"Confidence: {wasting_confidence:.2f}%"
    )

    st.subheader("Recommendations")

    if stunting_result == "Normal":
        st.success(
            "Growth appears normal. Continue providing balanced nutrition and regular growth monitoring."
        )
    elif stunting_result == "Stunted":
        st.warning(
            "Monitor growth closely. Increase protein-rich foods such as eggs, fish, milk, tofu, and tempeh."
        )
    elif stunting_result == "Severely Stunted":
        st.error(
            "High stunting risk detected. Seek professional healthcare advice and improve nutritional intake immediately."
        )
    elif stunting_result == "Tall":
        st.info(
            "Height is above the expected range for age."
        )
    if wasting_result == "Normal weight":
        st.success(
            "Weight appears appropriate for height."
        )
    elif wasting_result == "Underweight":
        st.warning(
            "Increase calorie and protein intake. Monitor weight regularly."
        )
    elif wasting_result == "Severely Underweight":
        st.error(
            "Severe underweight status detected. Professional medical assessment is recommended."
        )
    elif wasting_result == "Risk of Overweight":
        st.info(
            "Monitor dietary habits and physical activity levels."
        )
    
    st.markdown("---")

    st.caption(
        "Disclaimer: NutriVision is intended as a preliminary screening tool "
        "to assist in nutritional assessment. It does not replace professional "
        "medical advice, diagnosis, or treatment. Please consult a qualified "
        "healthcare professional for an accurate evaluation of a child's "
        "nutritional status."
    )