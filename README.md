Here is a complete, professional `README.md` file for your project. You can copy this directly into your GitHub repository or project folder.

It covers everything we have done: from the raw data to the final "SafeRoute Pro" web application.

-----

# 🛡️ SafeRoute Pro: Road Accident Severity Prediction

   

**SafeRoute Pro** is an AI-powered system designed to predict the severity of road accidents (**Slight**, **Serious**, or **Fatal**) based on environmental conditions, driver demographics, and vehicle characteristics. By leveraging Machine Learning, this tool assists emergency responders and policy-makers in assessing risk levels in real-time.

-----

## 📖 Table of Contents

1.  [Project Overview](https://www.google.com/search?q=%23-project-overview)
2.  [Key Features](https://www.google.com/search?q=%23-key-features)
3.  [Tech Stack](https://www.google.com/search?q=%23-tech-stack)
4.  [Dataset & Preprocessing](https://www.google.com/search?q=%23-dataset--preprocessing)
5.  [Model Architecture](https://www.google.com/search?q=%23-model-architecture)
6.  [Installation & Usage](https://www.google.com/search?q=%23-installation--usage)
7.  [Project Structure](https://www.google.com/search?q=%23-project-structure)
8.  [Results & Evaluation](https://www.google.com/search?q=%23-results--evaluation)
9.  [Contributors](https://www.google.com/search?q=%23-contributors)

-----

## 🔍 Project Overview

Road accidents are a leading cause of global mortality. A major challenge in emergency response is the lack of immediate context regarding an accident's severity.

**The Goal:** To build a predictive model that classifies injury risk using historical traffic data, prioritizing the detection of rare but deadly "Fatal" accidents.

**The Solution:** An end-to-end pipeline that processes raw accident data, trains a **Random Forest Classifier**, and deploys it via a user-friendly **Streamlit Web Application**.

-----

## 🌟 Key Features

  * **Glassmorphism UI:** A modern, responsive web interface with smooth transitions and a dark-mode aesthetic.
  * **Real-Time Prediction:** Instant risk assessment based on 30+ input variables.
  * **Safety-First Logic:** Implements **custom probability thresholds** (lowered to 10% for fatal detection) to prioritize safety (Recall) over raw accuracy.
  * **Visual Risk Indicators:** \* 🔴 **Critical Risk (Fatal)**
      * 🟠 **High Risk (Serious)**
      * 🔵 **Low Risk (Slight)**

-----

## 🛠 Tech Stack

  * **Language:** Python 3.9+
  * **Data Manipulation:** Pandas, NumPy
  * **Machine Learning:** Scikit-Learn (RandomForest, Pipeline, LabelEncoder)
  * **Model Persistence:** Joblib
  * **Web Framework:** Streamlit
  * **Visualization:** Matplotlib, Seaborn (for analysis)

-----

## 📊 Dataset & Preprocessing

The project uses the **RTA (Road Traffic Accident) Dataset**, containing \~12,000 records.

### 1\. Data Cleaning

  * **Missing Values:** Handled `na` values using **Mode Imputation** for categorical features.
  * **Feature Engineering:** Extracted `Hour_of_Day` from raw timestamps to capture rush-hour patterns.

### 2\. Encoding

  * Converted categorical text (e.g., "Wet", "Monday", "Male") into machine-readable numbers using `LabelEncoder` and `OrdinalEncoder`.

### 3\. Handling Imbalance

  * **The Problem:** 85% of data was "Slight Injury", causing the model to ignore "Fatal" cases.
  * **The Fix:** Used `class_weight='balanced'` in the Random Forest model to penalize misclassifications of the minority class.

-----

## 🤖 Model Architecture

We selected the **Random Forest Classifier** for its robustness against overfitting and ability to handle mixed data types.

### Configuration

```python
RandomForestClassifier(
    n_estimators=200,          # 200 Decision Trees
    class_weight='balanced',   # Handle Imbalanced Data automatically
    random_state=42
)
```

### Custom Thresholding (The "Secret Sauce")

Standard models predict "Fatal" only if confidence \> 50%. To save lives, we lowered the bar:

  * **Fatal Threshold:** \> 10% Probability
  * **Serious Threshold:** \> 20% Probability

*Result: significantly improved the Recall for high-severity accidents.*

-----

## 💻 Installation & Usage

### Prerequisites

Ensure you have Python installed. Clone this repository and install dependencies:

```bash
git clone https://github.com/YourUsername/SafeRoute-Pro.git
cd SafeRoute-Pro
pip install pandas numpy scikit-learn streamlit joblib matplotlib seaborn
```

### Running the App

To launch the "SafeRoute Pro" dashboard:

```bash
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

-----

## 📂 Project Structure

```text
SafeRoute-Pro/
│
├── data/
│   └── RTA Dataset.csv         # Original Raw Data
│
├── notebooks/
│   ├── 01_Data_Cleaning.ipynb  # Handling missing values & formatting
│   ├── 02_Visualization.ipynb  # EDA charts & graphs
│   ├── 03_Preprocessing.ipynb  # Encoding & Feature Engineering
│   └── 04_Modeling.ipynb       # Training Random Forest & Saving .joblib
│
├── app.py                      # The Streamlit Web Application (Frontend)
├── rta_model_pipeline.joblib   # Saved Trained Model
├── unique_values.joblib        # Saved Dropdown Options
└── README.md                   # Project Documentation
```

-----

## 📈 Results & Evaluation

  * **Overall Accuracy:** \~80%
  * **Key Insight:** The most significant predictors of severity were **Number of Casualties**, **Time of Day**, and **Vehicle Type**.
  * **Trade-off:** We accepted slightly lower overall precision to achieve higher **Recall** for fatal accidents, ensuring dangerous crashes are not missed.

-----

## 👥 Contributors

**Group 6 Capstone Team**

  * Louis-Marie Belfort Nzitongo Libero
  * Boakye Thomas Asamoah
  * Kelvin Anyimah
  * Emmanuel Kessie
  * Ransford Larbie

-----

*Built for the AI & Expert Systems Capstone Project.*
