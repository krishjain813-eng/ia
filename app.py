
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

# AUTO LOAD DATA
df = pd.read_csv("dataset.csv")

tabs = st.tabs(["Overview","Descriptive","Diagnostic","Predictive","Clustering","Prescriptive"])

with tabs[0]:
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Interested %", f"{df['Interested'].mean()*100:.1f}%")
    col4.metric("Avg Spending", int(df["loan_demand"].mean()))
    st.dataframe(df.head())

with tabs[1]:
    st.plotly_chart(px.histogram(df, x="revenue"), use_container_width=True)
    st.plotly_chart(px.box(df, y="payment_delay"), use_container_width=True)

with tabs[2]:
    st.write(pd.crosstab(df["cash_flow_issue"], df["Interested"]))

with tabs[3]:
    X = df.drop("Interested", axis=1)
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

    imp = pd.DataFrame({"Feature":X.columns,"Importance":model.feature_importances_})
    st.plotly_chart(px.bar(imp,x="Importance",y="Feature",orientation="h"), use_container_width=True)

with tabs[4]:
    kmeans = KMeans(n_clusters=3)
    df["cluster"] = kmeans.fit_predict(X)
    st.plotly_chart(px.scatter(df, x="revenue", y="loan_demand", color="cluster"), use_container_width=True)

with tabs[5]:
    df["strategy"] = np.where(df["Interested"]==1,"Offer Loan","Marketing")
    st.dataframe(df[["Interested","strategy"]].head())
