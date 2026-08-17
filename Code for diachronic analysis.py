#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:33:19 2026

@author: user
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================
# BERTopic Analysis of Climate-Related UNGA Sentences by Zone
# Input: all_climate_filtered_sentences.csv
# Output: Topic results, document-topic assignments, hierarchy files
# ============================================================

# Install first if needed:
# pip install bertopic sentence-transformers umap-learn hdbscan openpyxl

import os
import re
import pandas as pd

#Install if needed
#conda install -c conda-forge hdbscan umap-learn
#pip install bertopic sentence-transformers
#pip install plotly


from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

# ============================================================
# 1. FILE PATHS
# ============================================================

input_file = r"/Users/user/Desktop/Other Docs/UN/Diachronic/Data/all_climate_filtered_sentences.csv"

output_folder = r"/Users/user/Desktop/Other Docs/UN/Diachronic/BERTopic_Output"
os.makedirs(output_folder, exist_ok=True)

output_excel = os.path.join(output_folder, "BERTopic_Results_by_Zone.xlsx")


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(input_file)

print("Columns in dataset:")
print(df.columns)

required_columns = ["year", "Zone", "file", "label", "score", "sentence"]

missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in dataset: {missing_cols}")

df = df.dropna(subset=["sentence", "Zone"]).copy()
df["sentence"] = df["sentence"].astype(str)
df["Zone"] = df["Zone"].astype(str)


# ============================================================
# 3. TEXT CLEANING FUNCTION
# ============================================================

def clean_text(text):
    """
    Removes non-discursive markers, document separators, and excessive spacing.
    Adjust or expand the patterns depending on the structure of your corpus.
    """

    text = str(text)

    # Remove common document/header separators
    text = re.sub(r"={2,}", " ", text)
    text = re.sub(r"-{2,}", " ", text)
    text = re.sub(r"\*{2,}", " ", text)

    # Remove possible header labels
    text = re.sub(r"\b(session|meeting|agenda|speaker|president|assembly)\b\s*[:\-]", " ", text, flags=re.IGNORECASE)

    # Remove file-like markers
    text = re.sub(r"\b[A-Z]{3}_\d{1,3}_\d{4}\b", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove excessive spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["clean_sentence"] = df["sentence"].apply(clean_text)

df = df[df["clean_sentence"].str.len() > 20].copy()

print(f"Total valid climate-related sentences: {len(df)}")


# ============================================================
# 4. LOAD EMBEDDING MODEL
# ============================================================

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# 5. HELPER FUNCTION FOR DYNAMIC VECTORIZER SETTINGS
# ============================================================

def get_vectorizer_model(n_docs):
    """
    Dynamically adjusts document-frequency thresholds by zone size.
    """

    if n_docs < 50:
        min_df = 1
        max_df = 1.0
    elif n_docs < 200:
        min_df = 2
        max_df = 0.95
    elif n_docs < 1000:
        min_df = 3
        max_df = 0.90
    else:
        min_df = 5
        max_df = 0.85

    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=max_df
    )

    return vectorizer_model


# ============================================================
# 6. RUN BERTOPIC SEPARATELY FOR EACH ZONE
# ============================================================


all_topic_info = []
all_doc_topics = []
all_hierarchical_topics = []
run_summary = []

zones = sorted(df["Zone"].dropna().unique())

for zone in zones:

    print("\n" + "=" * 70)
    print(f"Running BERTopic for Zone: {zone}")
    print("=" * 70)

    zone_df = df[df["Zone"] == zone].copy()
    docs = zone_df["clean_sentence"].dropna().astype(str).tolist()

    n_docs = len(docs)
    print(f"Number of documents/sentences in zone: {n_docs}")

    if n_docs < 10:
        print(f"Skipping Zone {zone}: fewer than 10 valid documents.")
        run_summary.append({
            "Zone": zone,
            "Documents": n_docs,
            "Status": "Skipped",
            "Reason": "Fewer than 10 valid documents"
        })
        continue

    try:
        vectorizer_model = get_vectorizer_model(n_docs)

        topic_model = BERTopic(
            embedding_model=embedding_model,
            vectorizer_model=vectorizer_model,
            language="english",
            calculate_probabilities=False,
            verbose=True
        )

        topics, probabilities = topic_model.fit_transform(docs)

        # Topic information
        topic_info = topic_model.get_topic_info()
        topic_info["Zone"] = zone
        topic_info["Total_Documents_in_Zone"] = n_docs
        all_topic_info.append(topic_info)

        # Document-topic assignments
        zone_df["Topic"] = topics
        zone_df["Topic_Probability"] = None
        all_doc_topics.append(zone_df)

        # Hierarchical topics
        try:
            hierarchical_topics = topic_model.hierarchical_topics(docs)
            hierarchical_topics["Zone"] = zone
            all_hierarchical_topics.append(hierarchical_topics)

            fig = topic_model.visualize_hierarchy(
                hierarchical_topics=hierarchical_topics
            )

            html_file = os.path.join(
                output_folder,
                f"{str(zone).replace(' ', '_')}_topic_hierarchy.html"
            )

            fig.write_html(html_file)

        except Exception as e:
            print(f"Hierarchy failed for Zone {zone}: {e}")

        # Save model
        model_folder = os.path.join(
            output_folder,
            f"BERTopic_Model_{str(zone).replace(' ', '_')}"
        )

        topic_model.save(model_folder, serialization="safetensors")

        run_summary.append({
            "Zone": zone,
            "Documents": n_docs,
            "Status": "Completed",
            "Reason": ""
        })

        print(f"Completed Zone: {zone}")

    except Exception as e:
        print(f"BERTopic failed for Zone {zone}: {e}")

        run_summary.append({
            "Zone": zone,
            "Documents": n_docs,
            "Status": "Failed",
            "Reason": str(e)
        })


# ============================================================
# SAVE OUTPUT AFTER MODELLING
# ============================================================

with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:

    run_summary_df = pd.DataFrame(run_summary)
    run_summary_df.to_excel(writer, sheet_name="Run_Summary", index=False)

    if all_topic_info:
        combined_topic_info = pd.concat(all_topic_info, ignore_index=True)
        combined_topic_info.to_excel(writer, sheet_name="All_Topic_Info", index=False)

    if all_doc_topics:
        combined_doc_topics = pd.concat(all_doc_topics, ignore_index=True)
        combined_doc_topics.to_excel(writer, sheet_name="All_Doc_Topics", index=False)

    if all_hierarchical_topics:
        combined_hierarchy = pd.concat(all_hierarchical_topics, ignore_index=True)
        combined_hierarchy.to_excel(writer, sheet_name="All_Hierarchy", index=False)

print("\nDone.")
print(f"Results saved to:\n{output_excel}")

        # ====================================================
        # 6.1 TOPIC INFORMATION
        # ====================================================

        topic_info = topic_model.get_topic_info()
        topic_info["Zone"] = zone
        topic_info["Total_Documents_in_Zone"] = n_docs

        all_topic_info.append(topic_info)

        sheet_name = f"{zone}_Topic_Info"[:31]
        topic_info.to_excel(writer, sheet_name=sheet_name, index=False)

        # ====================================================
        # 6.2 DOCUMENT-TOPIC ASSIGNMENTS
        # ====================================================

        zone_df["Topic"] = topics

        if probabilities is not None:
            zone_df["Topic_Probability"] = [
                max(prob) if hasattr(prob, "__len__") else prob
                for prob in probabilities
            ]
        else:
            zone_df["Topic_Probability"] = None

        all_doc_topics.append(zone_df)

        sheet_name = f"{zone}_Doc_Topics"[:31]
        zone_df.to_excel(writer, sheet_name=sheet_name, index=False)

        # ====================================================
        # 6.3 TOPIC REPRESENTATIONS
        # ====================================================

        topic_terms = []

        for topic_id in topic_info["Topic"].tolist():

            if topic_id == -1:
                continue

            terms = topic_model.get_topic(topic_id)

            for rank, term_score in enumerate(terms, start=1):
                term, ctfidf_score = term_score

                topic_terms.append({
                    "Zone": zone,
                    "Topic": topic_id,
                    "Rank": rank,
                    "Term": term,
                    "cTFIDF_Score": ctfidf_score
                })

        topic_terms_df = pd.DataFrame(topic_terms)

        if not topic_terms_df.empty:
            sheet_name = f"{zone}_Topic_Terms"[:31]
            topic_terms_df.to_excel(writer, sheet_name=sheet_name, index=False)

        # ====================================================
        # 6.4 HIERARCHICAL TOPIC STRUCTURE
        # ====================================================

        try:
            hierarchical_topics = topic_model.hierarchical_topics(docs)
            hierarchical_topics["Zone"] = zone

            all_hierarchical_topics.append(hierarchical_topics)

            sheet_name = f"{zone}_Hierarchy"[:31]
            hierarchical_topics.to_excel(writer, sheet_name=sheet_name, index=False)

            # Save dendrogram as HTML
            fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
            html_file = os.path.join(output_folder, f"{zone}_topic_hierarchy.html")
            fig.write_html(html_file)

        except Exception as e:
            print(f"Could not generate hierarchy for Zone {zone}: {e}")

        # ====================================================
        # 6.5 SAVE MODEL FOR THIS ZONE
        # ====================================================

        model_folder = os.path.join(output_folder, f"BERTopic_Model_{zone}")
        topic_model.save(model_folder, serialization="safetensors")

        print(f"Completed Zone: {zone}")


    # ========================================================
    # 7. SAVE COMBINED OUTPUTS
    # ========================================================

    if all_topic_info:
        combined_topic_info = pd.concat(all_topic_info, ignore_index=True)
        combined_topic_info.to_excel(writer, sheet_name="All_Topic_Info", index=False)

    if all_doc_topics:
        combined_doc_topics = pd.concat(all_doc_topics, ignore_index=True)
        combined_doc_topics.to_excel(writer, sheet_name="All_Doc_Topics", index=False)

    if all_hierarchical_topics:
        combined_hierarchy = pd.concat(all_hierarchical_topics, ignore_index=True)
        combined_hierarchy.to_excel(writer, sheet_name="All_Hierarchy", index=False)


print("\nDone.")
print(f"Results saved to:\n{output_excel}")
print(f"HTML hierarchy files and saved BERTopic models are in:\n{output_folder}")


# ============================================================
# VADER SENTIMENT ANALYSIS BY ZONE
# ============================================================

# Install if needed:
pip install vaderSentiment openpyxl matplotlib

import os
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ============================================================
# 1. FILE PATHS
# ============================================================

input_file = r"/Users/user/Desktop/Other Docs/UN/Diachronic/Data/all_climate_filtered_sentences.csv"

output_folder = r"/Users/user/Desktop/Other Docs/UN/Diachronic/VADER_Output"
os.makedirs(output_folder, exist_ok=True)

output_excel = os.path.join(output_folder, "VADER_Sentiment_by_Zone.xlsx")
output_plot = os.path.join(output_folder, "VADER_Sentiment_Line_Graph_by_Zone.png")


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(input_file)

df = df.dropna(subset=["sentence", "Zone"]).copy()
df["sentence"] = df["sentence"].astype(str)
df["Zone"] = df["Zone"].astype(str)


# ============================================================
# 3. RUN VADER SENTIMENT ANALYSIS
# ============================================================

analyzer = SentimentIntensityAnalyzer()

def get_vader_scores(text):
    scores = analyzer.polarity_scores(text)
    return pd.Series({
        "VADER_Negative": scores["neg"],
        "VADER_Neutral": scores["neu"],
        "VADER_Positive": scores["pos"],
        "VADER_Compound": scores["compound"]
    })

vader_scores = df["sentence"].apply(get_vader_scores)

df_vader = pd.concat([df, vader_scores], axis=1)


# ============================================================
# 4. CLASSIFY SENTIMENT
# ============================================================

def classify_sentiment(compound):
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df_vader["VADER_Sentiment"] = df_vader["VADER_Compound"].apply(classify_sentiment)


# ============================================================
# 5. SUMMARISE SENTIMENT BY ZONE
# ============================================================

zone_summary = (
    df_vader
    .groupby("Zone")
    .agg(
        Number_of_Sentences=("sentence", "count"),
        Mean_Negative=("VADER_Negative", "mean"),
        Mean_Neutral=("VADER_Neutral", "mean"),
        Mean_Positive=("VADER_Positive", "mean"),
        Mean_Compound=("VADER_Compound", "mean")
    )
    .reset_index()
)

sentiment_counts = (
    df_vader
    .groupby(["Zone", "VADER_Sentiment"])
    .size()
    .reset_index(name="Count")
)

sentiment_percentages = sentiment_counts.copy()
sentiment_percentages["Percentage"] = (
    sentiment_percentages
    .groupby("Zone")["Count"]
    .transform(lambda x: (x / x.sum()) * 100)
)


# ============================================================
# 6. SAVE RESULTS TO EXCEL
# ============================================================

with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    df_vader.to_excel(writer, sheet_name="Sentence_Level_Scores", index=False)
    zone_summary.to_excel(writer, sheet_name="Zone_Summary", index=False)
    sentiment_counts.to_excel(writer, sheet_name="Sentiment_Counts", index=False)
    sentiment_percentages.to_excel(writer, sheet_name="Sentiment_Percentages", index=False)


# ============================================================
# 7. PRODUCE LINE GRAPH OF SENTIMENT RESULTS BY ZONE
# ============================================================

# Map zones to actual year ranges
zone_labels = {
    "Zone 1": "1985–1991",
    "Zone 2": "1992–1997",
    "Zone 3": "1998–2009",
    "Zone 4": "2010–2015",
    "Zone 5": "2016–2025"
}

# Create new label column
zone_summary["Time_Period"] = zone_summary["Zone"].map(zone_labels)

# Ensure correct chronological ordering
time_order = [
    "1985–1991",
    "1992–1997",
    "1998–2009",
    "2010–2015",
    "2016–2025"
]

zone_summary["Time_Period"] = pd.Categorical(
    zone_summary["Time_Period"],
    categories=time_order,
    ordered=True
)

zone_summary = zone_summary.sort_values("Time_Period")


# ============================================================
# LINE GRAPH
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    zone_summary["Time_Period"],
    zone_summary["Mean_Negative"],
    marker="o",
    linewidth=2,
    label="Negative"
)

plt.plot(
    zone_summary["Time_Period"],
    zone_summary["Mean_Neutral"],
    marker="o",
    linewidth=2,
    label="Neutral"
)

plt.plot(
    zone_summary["Time_Period"],
    zone_summary["Mean_Positive"],
    marker="o",
    linewidth=2,
    label="Positive"
)

plt.plot(
    zone_summary["Time_Period"],
    zone_summary["Mean_Compound"],
    marker="o",
    linewidth=2,
    label="Compound"
)

plt.xlabel("Diachronic Time Period")
plt.ylabel("Mean VADER Sentiment Score")
plt.title("VADER Sentiment Trends Across Diachronic UNGA Climate Discourse Periods")

plt.xticks(rotation=20)
plt.legend()

plt.tight_layout()

plt.savefig(output_plot, dpi=300)

plt.show()

print("VADER sentiment analysis completed.")
print(f"Excel results saved to:\n{output_excel}")
print(f"Line graph saved to:\n{output_plot}")


#GRAPH FOR COMPOUND SCORES ONLY

import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. FILE PATHS
# ============================================================

output_folder = r"/Users/user/Desktop/Other Docs/UN/Diachronic/VADER_Output"

# Existing VADER results file
output_excel = os.path.join(
    output_folder,
    "VADER_Sentiment_by_Zone.xlsx"
)

# New figure containing only compound scores
compound_plot = os.path.join(
    output_folder,
    "VADER_Compound_Score_by_Zone.png"
)


# ============================================================
# 2. LOAD EXISTING ZONE-LEVEL VADER RESULTS
# ============================================================

zone_summary = pd.read_excel(
    output_excel,
    sheet_name="Zone_Summary"
)


# ============================================================
# 3. MAP ZONES TO DIACHRONIC TIME PERIODS
# ============================================================

zone_labels = {
    "Zone 1": "1985–1991",
    "Zone 2": "1992–1997",
    "Zone 3": "1998–2009",
    "Zone 4": "2010–2015",
    "Zone 5": "2016–2025"
}

zone_summary["Time_Period"] = zone_summary["Zone"].map(zone_labels)


# ============================================================
# 4. ENSURE CORRECT CHRONOLOGICAL ORDER
# ============================================================

time_order = [
    "1985–1991",
    "1992–1997",
    "1998–2009",
    "2010–2015",
    "2016–2025"
]

zone_summary["Time_Period"] = pd.Categorical(
    zone_summary["Time_Period"],
    categories=time_order,
    ordered=True
)

zone_summary = zone_summary.sort_values("Time_Period")


# ============================================================
# 5. PLOT MEAN COMPOUND SCORE ONLY
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    zone_summary["Time_Period"],
    zone_summary["Mean_Compound"],
    marker="o",
    linewidth=2
)

# Reference line separating positive and negative polarity
plt.axhline(
    y=0,
    linestyle="--",
    linewidth=1
)

plt.xlabel("Diachronic Time Period")
plt.ylabel("Mean VADER Compound Score")

plt.title(
    "VADER Compound Sentiment Across Diachronic "
    "UNGA Climate Discourse Periods"
)

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    compound_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 6. DISPLAY VALUES USED IN THE FIGURE
# ============================================================

print(
    zone_summary[
        ["Time_Period", "Mean_Compound"]
    ].to_string(index=False)
)

print("\nCompound sentiment graph saved to:")
print(compound_plot)




