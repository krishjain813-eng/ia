
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

st.title("Finance Customer Intelligence Dashboard")

uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    # Simple visualization
    fig = px.histogram(df, x="Revenue")
    st.plotly_chart(fig)

    # Model
    X = df[["Payment_Delay","Cash_Flow_Issue","Digital_User","Loan_Demand"]]
    y = df["Interested"]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    st.subheader("Model Performance")
    st.write("Accuracy:", accuracy_score(y_test, preds))
    st.write("Precision:", precision_score(y_test, preds))
    st.write("Recall:", recall_score(y_test, preds))
    st.write("F1 Score:", f1_score(y_test, preds))
