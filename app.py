import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')
st.title("🚢 Titanic Dataset Exploratory Data Analysis")

@st.cache_data
def load_data():
    return pd.read_csv('titanic.csv')

df = load_data()

if st.checkbox("Show raw data"):
    st.write(df.head())

st.subheader("🔍 Missing Values")
st.write(df.isnull().sum())

st.subheader("📊 Age Distribution")
fig, ax = plt.subplots()
sns.histplot(df['Age'], bins=30, kde=True, ax=ax, color='teal')
st.pyplot(fig)

st.subheader("💰 Fare Box Plot")
fig, ax = plt.subplots()
sns.boxplot(x=df['Fare'], ax=ax)
st.pyplot(fig)

st.subheader("👥 Survival by Sex")
fig, ax = plt.subplots()
sns.countplot(x='Sex', hue='Survived', data=df, ax=ax)
st.pyplot(fig)

st.subheader("🔥 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)
