import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
data = pd.read_csv('jobs.csv')
data = data.drop("Unnamed: 0", axis=1)

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(input="content", stop_words="english")

# Extract all unique industries from the dataset
all_industries = data["Industry"].unique()

# Sort the industries alphabetically
all_industries = sorted(all_industries)

# Define function to filter skills based on selected industry
def get_skills_by_industry(industry):
    return data[data["Industry"] == industry]["Key Skills"].str.split(",").explode().unique()

# Define recommend_jobs function
def recommend_jobs(skills, data):
    skills = [skill.lower().strip() for skill in skills]
    matching_jobs = data[data["Key Skills"].str.lower().str.contains('|'.join(skills), regex=re.DOTALL)]
    tfidf_matrix = tfidf.fit_transform(matching_jobs["Key Skills"])
    similarity = cosine_similarity(tfidf_matrix)
    matching_jobs["Similarity"] = similarity.sum(axis=1)  # Sum of similarities
    matching_jobs = matching_jobs.sort_values(by="Similarity", ascending=False)

    # Aggregate jobs with the same title
    aggregated_jobs = matching_jobs.groupby("Job Title").first().reset_index()

    return aggregated_jobs[["Job Title", "Job Experience Required", "Key Skills"]]

# Streamlit UI
st.title("Job Recommendation System")
st.sidebar.title("Input Information")

# Dropdown menu for selecting industry
selected_industry = st.sidebar.selectbox("Select Industry", all_industries)

# Filter skills based on selected industry
if selected_industry:
    industry_skills = get_skills_by_industry(selected_industry)
    selected_skills = st.sidebar.multiselect("Select skills", industry_skills)

    if st.sidebar.button("Recommend Jobs"):
        if selected_skills:
            recommended_jobs = recommend_jobs(selected_skills, data)
            st.subheader("Recommended Jobs")
            st.dataframe(recommended_jobs)
        else:
            st.error("Please select some skills.")
