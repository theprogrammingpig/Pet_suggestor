import os
import joblib
import pandas as pd
import streamlit as st

# find the model file in the same folder as this app.py
folder = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(folder, "pet_recommender.joblib"))

columns = ["Aquarium_size", "Experience", "Animal_category",
           "Lighting", "Tank_liters", "Temperature_c"]

st.title("Pet Recommender")
st.write("Choose your setup and the model will suggest animals that fit.")

# each menu shows words; the dictionary turns the choice into the number the model learned
size = {"Small": 0, "Big": 1}
experience = {"Beginner": 0, "Intermediate": 1, "Expert": 2}
category = {"Arachnid": 0, "Freshwater fish": 1, "Saltwater animal": 2, "Invertebrate": 3,
            "Reptile": 4, "Amphibian": 5, "Insect": 6}
lighting = {"Low": 0, "Moderate": 1, "High": 2}

size_choice = st.radio("Tank size", list(size), horizontal=True)
exp_choice = st.radio("Your experience", list(experience), horizontal=True)
cat_choice = st.selectbox("Animal category", list(category))
light_choice = st.radio("Lighting", list(lighting), horizontal=True)
liters = st.select_slider("Tank volume (liters)",
                          options=[10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 300, 450, 600, 1000, 2000],
                          value=40)
temperature = st.slider("Temperature (°C)", 10, 40, 24)

if st.button("Find animals"):
    user_choice = pd.DataFrame([[size[size_choice], experience[exp_choice], category[cat_choice],
                                 lighting[light_choice], liters, temperature]], columns=columns)

    probabilities = model.predict_proba(user_choice)[0]
    results = pd.Series(probabilities, index=model.classes_)
    results = results[results > 0].sort_values(ascending=False).head(5)

    st.subheader("Best matches")
    for name, chance in results.items():
        st.write(f"**{name}** ({chance:.0%} match)")
