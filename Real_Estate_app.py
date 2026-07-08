
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Real Estate Investment Advisor", layout="wide")

st.title("🏠 Real Estate Investment Advisor")
st.write("Predict Good Investment and Future Property Price")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("india_housing_prices.csv")
    df = df.drop_duplicates()
    df.columns = df.columns.str.strip()

    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    df["Age_of_Property"] = 2026 - df["Year_Built"]

    median_price_sqft = df["Price_per_SqFt"].median()
    df["Good_Investment"] = (df["Price_per_SqFt"] <= median_price_sqft).astype(int)

    df["Future_Price_5Y"] = df["Price_in_Lakhs"] * ((1 + 0.08) ** 5)

    return df

df = load_data()

# -------------------------------
# ENCODING FOR FEATURE COLUMNS
# -------------------------------
drop_cols = ["ID"]
df_model = df.drop(columns=drop_cols, errors="ignore")

df_encoded = pd.get_dummies(df_model, drop_first=True)

X = df_encoded.drop(
    ["Good_Investment", "Future_Price_5Y"],
    axis=1
)

feature_columns = pickle.load(open("feature_columns.pkl", "rb"))
# -------------------------------
# LOAD SAVED MODELS
# -------------------------------
@st.cache_resource
def load_models():
    clf = pickle.load(open("classification_model.pkl", "rb"))
    reg = pickle.load(open("regression_model.pkl", "rb"))
    return clf, reg

clf, reg = load_models()

st.success("Models loaded successfully!")

# -------------------------------
# DATA PREVIEW
# -------------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# EDA
# -------------------------------
st.subheader("EDA Visualizations")

chart = st.selectbox(
    "Select Chart",
    [
        "Price Distribution",
        "Size vs Price",
        "Average Price by City",
        "BHK Distribution",
        "Correlation Heatmap"
    ]
)

if chart == "Price Distribution":
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["Price_in_Lakhs"], kde=True, ax=ax)
    ax.set_title("Property Price Distribution")
    st.pyplot(fig)

elif chart == "Size vs Price":
    fig, ax = plt.subplots(figsize=(8, 5))
    sample_chart = df.sample(n=min(5000, len(df)), random_state=42)
    sns.scatterplot(
        x="Size_in_SqFt",
        y="Price_in_Lakhs",
        data=sample_chart,
        ax=ax
    )
    ax.set_title("Size vs Price")
    st.pyplot(fig)

elif chart == "Average Price by City":
    city_price = (
        df.groupby("City")["Price_in_Lakhs"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    city_price.plot(kind="bar", ax=ax)
    ax.set_title("Top 10 Cities by Average Property Price")
    ax.set_ylabel("Average Price in Lakhs")
    st.pyplot(fig)

elif chart == "BHK Distribution":
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x="BHK", data=df, ax=ax)
    ax.set_title("BHK Distribution")
    st.pyplot(fig)

elif chart == "Correlation Heatmap":
    numeric_df = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

# -------------------------------
# INPUT FORM
# -------------------------------
st.header("Property Prediction")

col1, col2 = st.columns(2)

with col1:
    state = st.selectbox("State", sorted(df["State"].unique()))
    city = st.selectbox("City", sorted(df[df["State"] == state]["City"].unique()))
    current_price = st.number_input("Current Price in Lakhs", min_value=1.0, value=50.0)
    area = st.number_input("Size in SqFt", min_value=100, value=1000)
    bhk = st.number_input("BHK", min_value=1, value=2)

with col2:
    price_sqft = st.number_input(
        "Price per SqFt",
        min_value=0.0,
        value=0.10,
        step=0.01,
        format="%.2f"
    )

    property_type = st.selectbox("Property Type", sorted(df["Property_Type"].unique()))
    furnished_status = st.selectbox("Furnished Status", sorted(df["Furnished_Status"].unique()))
    age = st.number_input("Age of Property", min_value=0, value=5)
    floor_no = st.number_input("Floor Number", min_value=0, value=1)
    total_floors = st.number_input("Total Floors", min_value=1, value=5)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict"):
    input_data = {col: 0 for col in feature_columns}

    if "Price_in_Lakhs" in input_data:
        input_data["Price_in_Lakhs"] = current_price

    if "Size_in_SqFt" in input_data:
        input_data["Size_in_SqFt"] = area

    if "BHK" in input_data:
        input_data["BHK"] = bhk

    if "Price_per_SqFt" in input_data:
        input_data["Price_per_SqFt"] = price_sqft

    if "Age_of_Property" in input_data:
        input_data["Age_of_Property"] = age

    if "Floor_No" in input_data:
        input_data["Floor_No"] = floor_no

    if "Total_Floors" in input_data:
        input_data["Total_Floors"] = total_floors

    state_col = f"State_{state}"
    if state_col in input_data:
        input_data[state_col] = 1

    city_col = f"City_{city}"
    if city_col in input_data:
        input_data[city_col] = 1

    property_col = f"Property_Type_{property_type}"
    if property_col in input_data:
        input_data[property_col] = 1

    furnished_col = f"Furnished_Status_{furnished_status}"
    if furnished_col in input_data:
        input_data[furnished_col] = 1

    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_columns]

    investment_pred = clf.predict(input_df)[0]
    future_price = reg.predict(input_df)[0]

    st.subheader("Prediction Result")

    if investment_pred == 1:
        st.success("✅ Good Investment")
    else:
        st.error("❌ Not a Good Investment")

    st.info(f"Estimated Price After 5 Years: ₹ {future_price:,.2f} Lakhs")

st.markdown("---")
st.write("Real Estate Investment Advisor Project")
