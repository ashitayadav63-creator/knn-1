import streamlit as st
import numpy as np
import plotly.express as px
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

st.set_page_config(page_title="KNN Weather Classifier", layout="centered")

st.title("🌦️ KNN Weather Classification App")
st.write("Predict whether the weather is **Sunny ☀️** or **Rainy 🌧️** using KNN.")

# Dataset
X = np.array([[30, 70], [25, 80], [27, 60], [31, 65], [23, 85], [28, 75]])
y = np.array([0, 0, 0, 1, 1, 1])  # 0 = Sunny, 1 = Rainy

# Convert to DataFrame for visualization
df = pd.DataFrame(X, columns=["Temperature", "Humidity"])
df["Weather"] = y
df["Weather Label"] = df["Weather"].map({0: "Sunny", 1: "Rainy"})

# Sidebar inputs
st.sidebar.header("Input Weather Conditions")

temperature = st.sidebar.slider("Temperature (°C)", 20, 40, 26)
humidity = st.sidebar.slider("Humidity (%)", 50, 100, 78)
k_value = st.sidebar.slider("Number of Neighbors (K)", 1, 5, 3)

# Model
knn = KNeighborsClassifier(n_neighbors=k_value)
knn.fit(X, y)

new_point = np.array([[temperature, humidity]])
prediction = knn.predict(new_point)[0]

# Prediction Result
st.subheader("Prediction Result")

if prediction == 0:
    st.success("Predicted Weather: Sunny ☀️")
else:
    st.info("Predicted Weather: Rainy 🌧️")

# Add new point to visualization
new_df = pd.DataFrame(
    [[temperature, humidity, prediction,
      "Sunny" if prediction == 0 else "Rainy"]],
    columns=["Temperature", "Humidity", "Weather", "Weather Label"]
)

plot_df = pd.concat([df, new_df], ignore_index=True)

# Plotly Visualization
st.subheader("Visualization")

fig = px.scatter(
    plot_df,
    x="Temperature",
    y="Humidity",
    color="Weather Label",
    size=[15]*len(df) + [25],
    title="KNN Weather Classification",
)

# Highlight new point
fig.update_traces(marker=dict(line=dict(width=2, color='black')))

st.plotly_chart(fig, use_container_width=True)
