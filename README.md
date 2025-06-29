# 🚢 Titanic Dataset Exploratory Data Analysis with Streamlit

This project provides an **interactive EDA (Exploratory Data Analysis)** interface using **Streamlit**, allowing users to explore the Titanic dataset visually and gain insights into passenger survival patterns.

---

## 📌 Project Overview

The Titanic dataset is a classic machine learning dataset provided by Kaggle. It contains information on passengers aboard the Titanic, such as age, gender, passenger class, fare, and whether they survived.

This Streamlit app provides:
- Summary of missing values
- Visual distribution of age and fare
- Survival analysis by gender and passenger class
- Outlier detection using box plots
- Correlation heatmap of numerical variables

---

## 📊 Features

✅ Interactive visualization using **Seaborn + Matplotlib**  
✅ Auto-generated plots and data summaries  
✅ Simple toggle to show/hide raw data  
✅ Responsive layout for large screens  

---

## 🧠 Key Visuals Included

- **Histogram**: Age distribution
- **Box Plot**: Fare outlier detection
- **Count Plots**: Survival by Gender, Survival by Class
- **Heatmap**: Correlation between numerical variables

---

## 📂 Dataset

The app uses the `train.csv` file from the [Kaggle Titanic Challenge](https://www.kaggle.com/competitions/titanic). Make sure to rename the file as `titanic.csv` and place it in the root folder.

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/titanic-eda-streamlit.git
cd titanic-eda-streamlit
