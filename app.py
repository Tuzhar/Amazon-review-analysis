# AMAZON REVIEW ANALYZER
# -----------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from textblob import TextBlob
import numpy as np
import nltk
import plotly.express as px
import kaleido
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from wordcloud import WordCloud
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# ------------ 1. Loading Data -----------
df = pd.read_csv('Reviews_initiall.csv')
df = df[['Id', 'Text', 'Score']].dropna()
df = df.head(500)

# ----------- 2. Initializing Models ---------
sia = SentimentIntensityAnalyzer()
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def polarity_scores_roberta(text):
    encoded_text = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
    output = model(**encoded_text)
    scores = softmax(output[0][0].detach().numpy())
    return {'roberta_neg': scores[0], 'roberta_neu': scores[1], 'roberta_pos': scores[2]}

# ------------3. Analyzing Reviews -------------
res = {}
for i, row in df.iterrows():
    text = str(row['Text'])
    myid = row['Id']
    vader_result = sia.polarity_scores(text)
    vader_result = {f"vader_{k}": v for k, v in vader_result.items()}
    roberta_result = polarity_scores_roberta(text)
    res[myid] = {**vader_result, **roberta_result}

results_df = pd.DataFrame(res).T.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, on='Id', how='left')

# TextBlob sentiment
results_df['textblob_polarity'] = results_df['Text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
results_df['textblob_subjectivity'] = results_df['Text'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)

# Metrics and stuff
avg_scores = results_df[['roberta_neg','roberta_neu','roberta_pos']].mean()
positive_pct = avg_scores['roberta_pos'] * 100
neutral_pct = avg_scores['roberta_neu'] * 100
negative_pct = avg_scores['roberta_neg'] * 100
avg_polarity = results_df['textblob_polarity'].mean()
avg_subjectivity = results_df['textblob_subjectivity'].mean()
avg_weighted = ((results_df['roberta_pos'] + (results_df['textblob_polarity'] + 1)/2) / 2).mean()

if avg_weighted > 0.65:
    recommendation = "✅ Recommended to BUY"
elif avg_weighted < 0.35:
    recommendation = "❌ Not Recommended"
else:
    recommendation = "⚖️ Neutral / Mixed Reviews"

#  4. Streamlit Layout
st.set_page_config(page_title="Amazon Review Analyzer", layout="wide")
st.title("🧠 Amazon Review Analyzer")
# after st.title
st.markdown("### 🛍️ Analyze Amazon product reviews using advanced NLP models like RoBERTa, VADER, and TextBlob. \
Visualize sentiments, explore insights, and get data-backed product recommendations.")


# Sidebar filters
st.sidebar.header("🔎 Filters")
selected_rating = st.sidebar.multiselect("Star Ratings:", options=sorted(results_df['Score'].unique()),
                                         default=sorted(results_df['Score'].unique()))
filtered_df = results_df[results_df['Score'].isin(selected_rating)]

sentiment_filter = st.sidebar.radio("Sentiment Filter:", ["All", "Positive", "Neutral", "Negative"])
if sentiment_filter != "All":
    if sentiment_filter == "Positive":
        filtered_df = filtered_df[filtered_df['roberta_pos'] > 0.6]
    elif sentiment_filter == "Neutral":
        filtered_df = filtered_df[(filtered_df['roberta_pos'] <= 0.6) & (filtered_df['roberta_pos'] >= 0.4)]
    else:
        filtered_df = filtered_df[filtered_df['roberta_pos'] < 0.4]

st.markdown("### 📊 Key Metrics")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Reviews", len(filtered_df), delta=None)
col2.metric("Positive %", f"{positive_pct:.1f}%")
col3.metric("Neutral %", f"{neutral_pct:.1f}%")
col4.metric("Avg Polarity", f"{avg_polarity:.2f}")
col5.metric("Recommendation", recommendation)

st.markdown("---")

# Tabs for organization
tab1, tab2, tab3 = st.tabs(["Summary & Keywords", "Visual Insights", "Explore Data"])

# Tab 1: Summary
with tab1:
    st.subheader("📖 Overall Summary")
    summary_text = f"Reviews are {'highly positive' if positive_pct > 60 else 'mixed' if positive_pct > 40 else 'mostly negative'}, with ~{positive_pct:.1f}% positive and {negative_pct:.1f}% negative sentiment overall."
    st.info(summary_text)

    pos_words = [w for w in ['quality','taste','easy','good','value','recommend','perfect'] if w in " ".join(filtered_df['Text'].astype(str)).lower()]
    neg_words = [w for w in ['bad','late','broken','fake','poor','refund','expensive'] if w in " ".join(filtered_df['Text'].astype(str)).lower()]

    col1, col2 = st.columns(2)
    col1.success("Top Positive Keywords")
    col1.write(", ".join(pos_words))
    col2.error("Top Negative Keywords")
    col2.write(", ".join(neg_words))

# Tab 2: Visual Insights
with tab2:
    st.subheader("📊 Interactive Charts")

    # Pie chart
    fig_pie = px.pie(values=[positive_pct, neutral_pct, negative_pct],
                     names=['Positive','Neutral','Negative'],
                     color=['Positive','Neutral','Negative'],
                     color_discrete_map={'Positive':'#2ecc71','Neutral':'#f1c40f','Negative':'#e74c3c'},
                     title="Overall Sentiment Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)

    # Bar chart: Average Positive by Score
    avg_pos_by_score = filtered_df.groupby('Score')['roberta_pos'].mean().reset_index()
    fig_bar = px.bar(avg_pos_by_score, x='Score', y='roberta_pos', text='roberta_pos', title="Average Positive Sentiment by Star Rating")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Scatter: Polarity vs Subjectivity
    fig_scatter = px.scatter(filtered_df, x='textblob_polarity', y='textblob_subjectivity', hover_data=['Text'],
                             color='roberta_pos', color_continuous_scale='blues', title="TextBlob Polarity vs Subjectivity")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Word Cloud
st.subheader("Word Clouds")
col1, col2 = st.columns(2)

positive_text = " ".join(filtered_df[filtered_df['roberta_pos'] > 0.6]['Text'].astype(str))
negative_text = " ".join(filtered_df[filtered_df['roberta_pos'] < 0.4]['Text'].astype(str))

#  Prevent crash if no reviews available
if positive_text.strip():
    wc_pos = WordCloud(width=400, height=250, background_color='white', colormap='Greens').generate(positive_text)
else:
    wc_pos = WordCloud(width=400, height=250, background_color='white').generate("No positive reviews found")

if negative_text.strip():
    wc_neg = WordCloud(width=400, height=250, background_color='white', colormap='Reds').generate(negative_text)
else:
    wc_neg = WordCloud(width=400, height=250, background_color='white').generate("No negative reviews found")

# Display Positive WordCloud
fig_wc_pos = plt.figure(figsize=(8, 4))
plt.imshow(wc_pos, interpolation='bilinear')
plt.axis('off')
col1.pyplot(fig_wc_pos)

# Display Negative WordCloud
fig_wc_neg = plt.figure(figsize=(8, 4))
plt.imshow(wc_neg, interpolation='bilinear')
plt.axis('off')
col2.pyplot(fig_wc_neg)

# Streamlit warnings
if not positive_text.strip():
    st.warning("⚠️ No positive reviews found in this filtered selection.")
if not negative_text.strip():
    st.warning("⚠️ No negative reviews found in this filtered selection.")

# Tab 3: Data Table
with tab3:
    st.subheader("🔍 Explore Individual Reviews")
    st.dataframe(filtered_df[['Id','Text','Score','roberta_pos','textblob_polarity','textblob_subjectivity']], use_container_width=True)

    review_id = st.slider("Select Review ID:", min_value=int(filtered_df['Id'].min()),
                          max_value=int(filtered_df['Id'].max()), value=int(filtered_df['Id'].min()))
    selected_review = filtered_df[filtered_df['Id'] == review_id]
    if not selected_review.empty:
        st.markdown("**Selected Review Details:**")
        st.write("**Review Text:**", selected_review['Text'].values[0])
        st.write("**Star Rating:**", selected_review['Score'].values[0])
        st.write("**Roberta Positive Sentiment:**", f"{selected_review['roberta_pos'].values[0]:.2f}")
        st.write("**TextBlob Polarity:**", f"{selected_review['textblob_polarity'].values[0]:.2f}")
        st.write("**TextBlob Subjectivity:**", f"{selected_review['textblob_subjectivity'].values[0]:.2f}")





st.markdown("---")
st.markdown(
    """
    **Made with curiosity by Tushar**  
    [GitHub](https://github.com/Tuzhar) | [LinkedIn](https://www.linkedin.com/in/tushar-singh-086644262/)
    """
)

