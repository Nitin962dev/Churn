import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="EDA Dashboard",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# CSS
# -----------------------------
st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
background:#0f172a;
color:white;
}

.card{
background:#1e293b;
padding:15px;
border-radius:15px;
box-shadow:0px 0px 10px black;
}

</style>
""",unsafe_allow_html=True)

# -----------------------------
# Load Dataset
# -----------------------------

@st.cache_data
def load_data():
    return pd.read_csv("Churn_Modelling.csv")

df = load_data()

# -----------------------------
# Title
# -----------------------------

st.title("📊 Exploratory Data Analysis")

st.write("---")

# -----------------------------
# Dataset Preview
# -----------------------------

st.subheader("Dataset")

st.dataframe(df, use_container_width=True)

# -----------------------------
# Shape
# -----------------------------

c1,c2,c3,c4=st.columns(4)

with c1:
    st.metric("Rows",df.shape[0])

with c2:
    st.metric("Columns",df.shape[1])

with c3:
    st.metric("Numerical",len(df.select_dtypes(include="number").columns))

with c4:
    st.metric("Categorical",len(df.select_dtypes(include="object").columns))

st.write("---")

# -----------------------------
# Missing Values
# -----------------------------

st.subheader("Missing Values")

missing=df.isnull().sum()

missing=missing[missing>0]

if len(missing)==0:

    st.success("No Missing Values")

else:

    st.dataframe(missing)

st.write("---")

# -----------------------------
# Data Types
# -----------------------------

st.subheader("Data Types")

dtype=pd.DataFrame(df.dtypes,columns=["Datatype"])

st.dataframe(dtype)

st.write("---")

# -----------------------------
# Numerical Summary
# -----------------------------

st.subheader("Statistical Summary")

st.dataframe(df.describe())

st.write("---")

# -----------------------------
# Histogram
# -----------------------------

st.subheader("Histogram")

num_cols=df.select_dtypes(include="number").columns

column=st.selectbox("Select Numerical Column",num_cols)

fig=px.histogram(
    df,
    x=column,
    color_discrete_sequence=["deepskyblue"],
    template="plotly_dark"
)

st.plotly_chart(fig,use_container_width=True)

# -----------------------------
# Boxplot
# -----------------------------

st.subheader("Box Plot")

box=px.box(
    df,
    y=column,
    color_discrete_sequence=["orange"],
    template="plotly_dark"
)

st.plotly_chart(box,use_container_width=True)

# -----------------------------
# Pie Chart
# -----------------------------

st.subheader("Pie Chart")

cat_cols=df.select_dtypes(include="object").columns

cat=st.selectbox("Select Categorical Column",cat_cols)

pie=px.pie(
    df,
    names=cat,
    hole=.4,
    template="plotly_dark"
)

st.plotly_chart(pie,use_container_width=True)

# -----------------------------
# Count Plot
# -----------------------------

st.subheader("Count Plot")

count=px.histogram(
    df,
    x=cat,
    color=cat,
    template="plotly_dark"
)

st.plotly_chart(count,use_container_width=True)

# -----------------------------
# Correlation
# -----------------------------

st.subheader("Correlation Heatmap")

corr=df.select_dtypes(include="number").corr()

fig=ff.create_annotated_heatmap(
    z=corr.values,
    x=list(corr.columns),
    y=list(corr.columns),
    annotation_text=round(corr,2).values,
    colorscale="Viridis"
)

fig.update_layout(
    template="plotly_dark",
    height=700
)

st.plotly_chart(fig,use_container_width=True)

# -----------------------------
# Scatter Plot
# -----------------------------

st.subheader("Scatter Plot")

c1,c2=st.columns(2)

with c1:

    x=st.selectbox(
        "X Axis",
        num_cols,
        key=1
    )

with c2:

    y=st.selectbox(
        "Y Axis",
        num_cols,
        key=2
    )

scatter=px.scatter(
    df,
    x=x,
    y=y,
    color="Exited",
    template="plotly_dark"
)

st.plotly_chart(scatter,use_container_width=True)

# -----------------------------
# Distribution
# -----------------------------

st.subheader("Distribution Plot")

dist=px.violin(
    df,
    y=column,
    box=True,
    template="plotly_dark"
)

st.plotly_chart(dist,use_container_width=True)

# -----------------------------
# Target Distribution
# -----------------------------

st.subheader("Target Variable")

target=px.histogram(
    df,
    x="Exited",
    color="Exited",
    template="plotly_dark"
)

st.plotly_chart(target,use_container_width=True)

# -----------------------------
# Download Dataset
# -----------------------------

csv=df.to_csv(index=False)

st.download_button(
    "⬇ Download Dataset",
    csv,
    "Churn_Modelling.csv",
    "text/csv"
)
