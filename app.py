
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.cluster import KMeans

st.set_page_config(layout="wide")

st.title("Finance Credit Analytics Studio")

file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if file:
    df = pd.read_csv(file)
else:
    st.stop()

tabs = st.tabs(["Overview","Descriptive","Predictive","Clustering"])

with tabs[0]:
    col1,col2,col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Interested %", f"{df['Interested'].mean()*100:.1f}%")
    st.dataframe(df.head())

with tabs[1]:
    fig = px.histogram(df, x="revenue")
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    X = df.select_dtypes(include=np.number).drop("Interested",axis=1)
    y = df["Interested"]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X_train,y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]
    st.write("Accuracy:", accuracy_score(y_test,preds))
    st.write("Precision:", precision_score(y_test,preds))
    st.write("Recall:", recall_score(y_test,preds))
    st.write("F1:", f1_score(y_test,preds))
    st.write("ROC AUC:", roc_auc_score(y_test,probs))

with tabs[3]:
    kmeans = KMeans(n_clusters=3)
    df["cluster"] = kmeans.fit_predict(X)
    fig = px.scatter(df, x="revenue", y="loan_demand", color="cluster")
    st.plotly_chart(fig, use_container_width=True)
