# ============================================================================
# sentiment_analysis.py -- Main Analysis Script
# Employee Sentiment Analysis Project
# ============================================================================
# This script implements all 6 tasks from the project specification:
#   Task 1: Sentiment Labeling
#   Task 2: Exploratory Data Analysis (EDA)
#   Task 3: Employee Score Calculation
#   Task 4: Employee Ranking
#   Task 5: Flight Risk Identification
#   Task 6: Predictive Modeling
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from collections import defaultdict
import os
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set visual style
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
sns.set_theme(style='whitegrid', palette='husl')

# Create visualization directory
VIZ_DIR = 'visualization'
os.makedirs(VIZ_DIR, exist_ok=True)

# ============================================================================
# PHASE 0: DATA LOADING & CLEANING
# ============================================================================
from data_prep import load_and_clean_data

df = load_and_clean_data('test(in).csv')

print(f"\n{'='*60}")
print("Data loaded successfully. Starting analysis pipeline...")
print(f"{'='*60}")


# ============================================================================
# TASK 1: SENTIMENT LABELING
# ============================================================================
print(f"\n{'='*60}")
print("TASK 1: Sentiment Labeling")
print(f"{'='*60}")

print("""
APPROACH & JUSTIFICATION:
-------------------------
We use the 'cardiffnlp/twitter-roberta-base-sentiment-latest' model from
HuggingFace Transformers. This is a RoBERTa model fine-tuned on ~124M tweets 
for 3-class sentiment classification (Positive, Neutral, Negative).

Why this model?
  1. Native 3-class output -- matches our requirement exactly
  2. Pre-trained and well-validated -- no training data needed
  3. Handles informal/mixed-tone text well
  4. Runs locally -- fully reproducible, no API dependencies
  5. Widely cited in NLP research

The model processes each message's combined subject + body text and returns
the most probable sentiment class.
""")

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

print("[1/3] Loading sentiment model...")
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    truncation=True,
    device=-1  # CPU; change to 0 for GPU if available
)

# Map model labels to our labels
label_map = {
    'positive': 'Positive',
    'negative': 'Negative',
    'neutral': 'Neutral'
}

print("[2/3] Classifying sentiment for all messages...")
print("  (This may take several minutes for ~19K messages...)")

def classify_sentiment(text, max_chars=500):
    """Classify sentiment of a text using the transformer model."""
    if not text or len(str(text).strip()) == 0:
        return 'Neutral'
    
    # Truncate very long texts to first 500 chars for speed
    text = str(text)[:max_chars]
    
    try:
        result = sentiment_pipeline(text)[0]
        label = result['label'].lower()
        return label_map.get(label, 'Neutral')
    except Exception as e:
        return 'Neutral'

# Process in batches for efficiency
batch_size = 64
sentiments = []
total = len(df)

for i in range(0, total, batch_size):
    batch_texts = df['full_text'].iloc[i:i+batch_size].tolist()
    # Truncate texts for the pipeline
    batch_texts = [str(t)[:500] if t and len(str(t).strip()) > 0 else "neutral message" for t in batch_texts]
    
    try:
        results = sentiment_pipeline(batch_texts, batch_size=batch_size)
        for r in results:
            label = r['label'].lower()
            sentiments.append(label_map.get(label, 'Neutral'))
    except Exception as e:
        # Fallback: process individually
        for t in batch_texts:
            sentiments.append(classify_sentiment(t))
    
    if (i // batch_size) % 25 == 0:
        print(f"  Progress: {min(i + batch_size, total)}/{total} messages processed")

df['sentiment'] = sentiments

print(f"\n[3/3] Sentiment labeling complete!")
print(f"\nSentiment Distribution:")
print(df['sentiment'].value_counts().to_string())
print(f"\nVerification: NaN sentiments = {df['sentiment'].isna().sum()}")

# Save labeled dataset
df.to_csv('labeled_data.csv', index=False)
print("\nLabeled dataset saved to 'labeled_data.csv'")


# ============================================================================
# TASK 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print(f"\n{'='*60}")
print("TASK 2: Exploratory Data Analysis (EDA)")
print(f"{'='*60}")

# ------------------------------------------------------------------
# 2.1 Dataset Overview
# ------------------------------------------------------------------
print("\n-- 2.1 Dataset Overview --")
print(f"  Total records: {len(df)}")
print(f"  Columns: {list(df.columns)}")
print(f"\nData Types:")
print(df.dtypes.to_string())
print(f"\nMissing Values:")
print(df.isnull().sum().to_string())
print(f"\nBasic Statistics:")
print(df[['msg_length', 'word_count']].describe().to_string())

# ------------------------------------------------------------------
# 2.2 Sentiment Distribution
# ------------------------------------------------------------------
print("\n-- 2.2 Sentiment Distribution --")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart
colors = {'Positive': '#2ecc71', 'Neutral': '#3498db', 'Negative': '#e74c3c'}
sentiment_counts = df['sentiment'].value_counts()
bars = axes[0].bar(sentiment_counts.index, sentiment_counts.values, 
                   color=[colors.get(s, '#95a5a6') for s in sentiment_counts.index],
                   edgecolor='white', linewidth=1.5)
axes[0].set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Sentiment')
axes[0].set_ylabel('Count')
for bar, count in zip(bars, sentiment_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
                f'{count}\n({count/len(df)*100:.1f}%)', ha='center', va='bottom', fontweight='bold')

# Pie chart
axes[1].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
            colors=[colors.get(s, '#95a5a6') for s in sentiment_counts.index],
            startangle=90, explode=[0.05]*len(sentiment_counts),
            shadow=True, textprops={'fontsize': 12})
axes[1].set_title('Sentiment Proportion', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{VIZ_DIR}/sentiment_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: sentiment_distribution.png")

# ------------------------------------------------------------------
# 2.3 Sentiment Trends Over Time
# ------------------------------------------------------------------
print("\n-- 2.3 Sentiment Trends Over Time --")

monthly_sentiment = df.groupby([df['date'].dt.to_period('M'), 'sentiment']).size().unstack(fill_value=0)
monthly_sentiment.index = monthly_sentiment.index.to_timestamp()

fig, ax = plt.subplots(figsize=(14, 6))
for sent, color in colors.items():
    if sent in monthly_sentiment.columns:
        ax.plot(monthly_sentiment.index, monthly_sentiment[sent], 
                label=sent, color=color, linewidth=2, marker='o', markersize=4)

ax.set_title('Monthly Sentiment Trends', fontsize=14, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Message Count')
ax.legend(frameon=True, fancybox=True, shadow=True)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{VIZ_DIR}/sentiment_trends_over_time.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: sentiment_trends_over_time.png")

# ------------------------------------------------------------------
# 2.4 Top Senders by Volume
# ------------------------------------------------------------------
print("\n-- 2.4 Top Senders by Volume --")

top_senders = df['employee'].value_counts().head(15)

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(range(len(top_senders)), top_senders.values, color=sns.color_palette('viridis', len(top_senders)))
ax.set_yticks(range(len(top_senders)))
ax.set_yticklabels(top_senders.index)
ax.set_xlabel('Number of Messages')
ax.set_title('Top 15 Employees by Message Volume', fontsize=14, fontweight='bold')
ax.invert_yaxis()
for i, v in enumerate(top_senders.values):
    ax.text(v + 5, i, str(v), va='center', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{VIZ_DIR}/top_senders.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: top_senders.png")

# ------------------------------------------------------------------
# 2.5 Message Length Distribution
# ------------------------------------------------------------------
print("\n-- 2.5 Message Length Distribution --")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# By character length
axes[0].hist(df['msg_length'].clip(upper=5000), bins=50, color='#3498db', alpha=0.7, edgecolor='white')
axes[0].set_title('Message Length Distribution (chars)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Character Count')
axes[0].set_ylabel('Frequency')
axes[0].axvline(df['msg_length'].median(), color='red', linestyle='--', label=f"Median: {df['msg_length'].median():.0f}")
axes[0].legend()

# By word count
axes[1].hist(df['word_count'].clip(upper=1000), bins=50, color='#e67e22', alpha=0.7, edgecolor='white')
axes[1].set_title('Word Count Distribution', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Word Count')
axes[1].set_ylabel('Frequency')
axes[1].axvline(df['word_count'].median(), color='red', linestyle='--', label=f"Median: {df['word_count'].median():.0f}")
axes[1].legend()

plt.tight_layout()
plt.savefig(f'{VIZ_DIR}/message_length_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: message_length_distribution.png")

# ------------------------------------------------------------------
# 2.6 Message Length by Sentiment
# ------------------------------------------------------------------
print("\n-- 2.6 Message Length by Sentiment --")

fig, ax = plt.subplots(figsize=(10, 6))
sentiment_order = ['Positive', 'Neutral', 'Negative']
palette = [colors[s] for s in sentiment_order if s in df['sentiment'].unique()]
existing_order = [s for s in sentiment_order if s in df['sentiment'].unique()]

sns.boxplot(data=df[df['word_count'] < 500], x='sentiment', y='word_count', 
            order=existing_order, palette=palette, ax=ax, showfliers=False)
ax.set_title('Word Count by Sentiment Category', fontsize=14, fontweight='bold')
ax.set_xlabel('Sentiment')
ax.set_ylabel('Word Count')
plt.tight_layout()
plt.savefig(f'{VIZ_DIR}/message_length_by_sentiment.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: message_length_by_sentiment.png")

# ------------------------------------------------------------------
# 2.7 Day-of-Week Analysis
# ------------------------------------------------------------------
print("\n-- 2.7 Day-of-Week Analysis --")

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_sentiment = df.groupby(['day_of_week', 'sentiment']).size().unstack(fill_value=0)
day_sentiment = day_sentiment.reindex(day_order)

fig, ax = plt.subplots(figsize=(12, 6))
day_sentiment.plot(kind='bar', stacked=True, ax=ax, 
                   color=[colors.get(c, '#95a5a6') for c in day_sentiment.columns],
                   edgecolor='white', linewidth=0.5)
ax.set_title('Sentiment Distribution by Day of Week', fontsize=14, fontweight='bold')
ax.set_xlabel('Day of Week')
ax.set_ylabel('Message Count')
ax.legend(title='Sentiment', frameon=True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{VIZ_DIR}/sentiment_by_day_of_week.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: sentiment_by_day_of_week.png")

# ------------------------------------------------------------------
# 2.8 Word Cloud per Sentiment
# ------------------------------------------------------------------
print("\n-- 2.8 Word Clouds --")

try:
    from wordcloud import WordCloud
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for idx, (sent, color_hex) in enumerate([('Positive', '#2ecc71'), ('Neutral', '#3498db'), ('Negative', '#e74c3c')]):
        text = ' '.join(df[df['sentiment'] == sent]['body_clean'].dropna().astype(str).tolist())
        if text.strip():
            wc = WordCloud(width=800, height=400, background_color='white',
                          max_words=100, colormap='Greens' if sent == 'Positive' else ('Blues' if sent == 'Neutral' else 'Reds'),
                          contour_width=2, contour_color=color_hex).generate(text)
            axes[idx].imshow(wc, interpolation='bilinear')
        axes[idx].set_title(f'{sent} Messages', fontsize=14, fontweight='bold')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{VIZ_DIR}/word_clouds.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: word_clouds.png")
except ImportError:
    print("  [SKIP] wordcloud not installed -- skipping word clouds")

# ------------------------------------------------------------------
# 2.9 Anomaly Detection: Message Volume Spikes
# ------------------------------------------------------------------
print("\n-- 2.9 Anomaly Detection --")

daily_volume = df.groupby(df['date'].dt.date).size()
mean_vol = daily_volume.mean()
std_vol = daily_volume.std()
anomaly_threshold = mean_vol + 2 * std_vol

anomaly_dates = daily_volume[daily_volume > anomaly_threshold]

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(daily_volume.index, daily_volume.values, color='#3498db', alpha=0.6, linewidth=0.8)
ax.axhline(y=mean_vol, color='green', linestyle='--', label=f'Mean ({mean_vol:.0f})')
ax.axhline(y=anomaly_threshold, color='red', linestyle='--', label=f'Anomaly threshold ({anomaly_threshold:.0f})')
if len(anomaly_dates) > 0:
    ax.scatter(anomaly_dates.index, anomaly_dates.values, color='red', s=50, zorder=5, label='Anomalies')
ax.set_title('Daily Message Volume with Anomaly Detection', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Message Count')
ax.legend(frameon=True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{VIZ_DIR}/anomaly_detection.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"  Anomaly threshold (mean + 2sigma): {anomaly_threshold:.0f} messages/day")
print(f"  Anomalous days detected: {len(anomaly_dates)}")
if len(anomaly_dates) > 0:
    print(f"  Top anomaly dates: {anomaly_dates.sort_values(ascending=False).head(5).to_string()}")
print("  Saved: anomaly_detection.png")


# ============================================================================
# TASK 3: EMPLOYEE SCORE CALCULATION
# ============================================================================
print(f"\n{'='*60}")
print("TASK 3: Employee Score Calculation")
print(f"{'='*60}")

print("""
SCORING METHOD:
---------------
  Positive → +1
  Neutral  →  0
  Negative → -1

Scores are aggregated per employee per calendar month.
The score resets at the start of each new month (no carry-over).
""")

# Map sentiment to score
score_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
df['score'] = df['sentiment'].map(score_map)

# Group by employee and year-month
monthly_scores = df.groupby(['employee', 'employee_email', 'year_month']).agg(
    total_score=('score', 'sum'),
    msg_count=('score', 'count'),
    positive_count=('score', lambda x: (x == 1).sum()),
    neutral_count=('score', lambda x: (x == 0).sum()),
    negative_count=('score', lambda x: (x == -1).sum())
).reset_index()

print("Monthly scores computed!")
print(f"\nSample (first 10 rows):")
print(monthly_scores.head(10).to_string(index=False))
print(f"\nTotal employee-month combinations: {len(monthly_scores)}")

# Save monthly scores
monthly_scores.to_csv('monthly_scores.csv', index=False)
print("Saved: monthly_scores.csv")


# ============================================================================
# TASK 4: EMPLOYEE RANKING
# ============================================================================
print(f"\n{'='*60}")
print("TASK 4: Employee Ranking")
print(f"{'='*60}")

print("""
RANKING METHOD:
---------------
For each month:
  - Top 3 Positive: employees with the HIGHEST monthly scores
  - Top 3 Negative: employees with the LOWEST (most negative) monthly scores
  - Ties broken alphabetically by employee name (ascending)
""")

all_top_positive = []
all_top_negative = []

unique_months = sorted(monthly_scores['year_month'].unique())

for ym in unique_months:
    month_data = monthly_scores[monthly_scores['year_month'] == ym].copy()
    
    # Sort: by score descending, then employee name ascending for ties
    month_data_sorted = month_data.sort_values(
        by=['total_score', 'employee'], ascending=[False, True]
    )
    
    top_3_pos = month_data_sorted.head(3)[['employee', 'year_month', 'total_score']].copy()
    top_3_pos['rank_type'] = 'Top Positive'
    top_3_pos['rank'] = range(1, len(top_3_pos) + 1)
    
    # Sort: by score ascending (most negative first), then employee name ascending for ties
    month_data_sorted_neg = month_data.sort_values(
        by=['total_score', 'employee'], ascending=[True, True]
    )
    
    top_3_neg = month_data_sorted_neg.head(3)[['employee', 'year_month', 'total_score']].copy()
    top_3_neg['rank_type'] = 'Top Negative'
    top_3_neg['rank'] = range(1, len(top_3_neg) + 1)
    
    all_top_positive.append(top_3_pos)
    all_top_negative.append(top_3_neg)

rankings_positive = pd.concat(all_top_positive, ignore_index=True)
rankings_negative = pd.concat(all_top_negative, ignore_index=True)

print("\n-- TOP 3 POSITIVE EMPLOYEES (per month) --")
for ym in unique_months:
    month_pos = rankings_positive[rankings_positive['year_month'] == ym]
    if len(month_pos) > 0:
        print(f"\n  {ym}:")
        for _, row in month_pos.iterrows():
            print(f"    #{int(row['rank'])} {row['employee']} (score: {row['total_score']})")

print("\n\n-- TOP 3 NEGATIVE EMPLOYEES (per month) --")
for ym in unique_months:
    month_neg = rankings_negative[rankings_negative['year_month'] == ym]
    if len(month_neg) > 0:
        print(f"\n  {ym}:")
        for _, row in month_neg.iterrows():
            print(f"    #{int(row['rank'])} {row['employee']} (score: {row['total_score']})")

# Visualization: Overall ranking across all months
print("\n-- Generating ranking visualizations --")

# Aggregate across all months for overall top positive
overall_scores = monthly_scores.groupby('employee')['total_score'].sum().sort_values(ascending=False)
top_10_pos_overall = overall_scores.head(10)
top_10_neg_overall = overall_scores.tail(10).sort_values()

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Top 10 Positive
bars_pos = axes[0].barh(range(len(top_10_pos_overall)), top_10_pos_overall.values, 
                         color=sns.color_palette('Greens_r', len(top_10_pos_overall)))
axes[0].set_yticks(range(len(top_10_pos_overall)))
axes[0].set_yticklabels(top_10_pos_overall.index)
axes[0].set_xlabel('Total Sentiment Score')
axes[0].set_title('Top 10 Most Positive Employees (Overall)', fontsize=13, fontweight='bold')
axes[0].invert_yaxis()
for i, v in enumerate(top_10_pos_overall.values):
    axes[0].text(v + 0.5, i, f'{v:.0f}', va='center', fontweight='bold')

# Top 10 Negative
bars_neg = axes[1].barh(range(len(top_10_neg_overall)), top_10_neg_overall.values,
                         color=sns.color_palette('Reds', len(top_10_neg_overall)))
axes[1].set_yticks(range(len(top_10_neg_overall)))
axes[1].set_yticklabels(top_10_neg_overall.index)
axes[1].set_xlabel('Total Sentiment Score')
axes[1].set_title('Top 10 Most Negative Employees (Overall)', fontsize=13, fontweight='bold')
axes[1].invert_yaxis()
for i, v in enumerate(top_10_neg_overall.values):
    axes[1].text(v - 0.5 if v < 0 else v + 0.5, i, f'{v:.0f}', va='center', fontweight='bold', ha='right' if v < 0 else 'left')

plt.tight_layout()
plt.savefig(f'{VIZ_DIR}/employee_rankings.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: employee_rankings.png")

# Monthly ranking bar chart for the most recent month
if len(unique_months) > 0:
    latest_month = unique_months[-1]
    latest_data = monthly_scores[monthly_scores['year_month'] == latest_month].sort_values('total_score', ascending=False)
    top_n = min(10, len(latest_data))
    
    fig, ax = plt.subplots(figsize=(12, 7))
    bar_colors = ['#2ecc71' if s > 0 else ('#e74c3c' if s < 0 else '#95a5a6') for s in latest_data.head(top_n)['total_score']]
    ax.barh(range(top_n), latest_data.head(top_n)['total_score'].values, color=bar_colors, edgecolor='white')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(latest_data.head(top_n)['employee'].values)
    ax.set_xlabel('Sentiment Score')
    ax.set_title(f'Employee Sentiment Scores -- {latest_month}', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.axvline(x=0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(f'{VIZ_DIR}/monthly_ranking_latest.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: monthly_ranking_latest.png")

# Save rankings
rankings_positive.to_csv('rankings_positive.csv', index=False)
rankings_negative.to_csv('rankings_negative.csv', index=False)
print("  Saved: rankings_positive.csv, rankings_negative.csv")


# ============================================================================
# TASK 5: FLIGHT RISK IDENTIFICATION
# ============================================================================
print(f"\n{'='*60}")
print("TASK 5: Flight Risk Identification")
print(f"{'='*60}")

print("""
FLIGHT RISK DEFINITION:
-----------------------
An employee is flagged as a flight risk if they sent 4 or more negative 
messages within ANY rolling 30-day window.

IMPLEMENTATION:
  - For each employee, collect all dates of negative messages (sorted)
  - Use a sliding window approach: for each negative message date, count
    how many other negative messages occurred within the next 30 days
  - If any window contains >= 4 negative messages, flag the employee
  - This handles overlapping windows correctly
""")

# Get all negative messages with dates
negative_msgs = df[df['sentiment'] == 'Negative'][['employee', 'employee_email', 'date']].copy()
negative_msgs = negative_msgs.sort_values(['employee', 'date'])

def check_flight_risk(dates, threshold=4, window_days=30):
    """
    Check if there exists any 30-day rolling window with >= threshold negative messages.
    Uses a sliding window approach on sorted dates.
    
    Returns:
        tuple: (is_at_risk: bool, max_count_in_window: int, window_start: date or None)
    """
    dates_sorted = sorted(dates)
    n = len(dates_sorted)
    
    if n < threshold:
        return False, n, None
    
    max_count = 0
    worst_window_start = None
    
    for i in range(n):
        window_end = dates_sorted[i] + pd.Timedelta(days=window_days)
        # Count messages in [dates_sorted[i], dates_sorted[i] + 30 days]
        count = sum(1 for d in dates_sorted[i:] if d <= window_end)
        if count > max_count:
            max_count = count
            worst_window_start = dates_sorted[i]
    
    return max_count >= threshold, max_count, worst_window_start

# Check each employee
flight_risk_results = []

for employee, group in negative_msgs.groupby('employee'):
    email = group['employee_email'].iloc[0]
    dates = group['date'].tolist()
    is_risk, max_neg, window_start = check_flight_risk(dates)
    
    flight_risk_results.append({
        'employee': employee,
        'employee_email': email,
        'total_negative_msgs': len(dates),
        'max_negatives_in_30d_window': max_neg,
        'is_flight_risk': is_risk,
        'worst_window_start': window_start
    })

flight_risk_df = pd.DataFrame(flight_risk_results)

# Filter flagged employees
flagged = flight_risk_df[flight_risk_df['is_flight_risk'] == True].sort_values(
    'max_negatives_in_30d_window', ascending=False
)

print(f"\nTotal employees analyzed: {len(flight_risk_df)}")
print(f"Employees flagged as FLIGHT RISK: {len(flagged)}")

print(f"\n-- FLIGHT RISK EMPLOYEES --")
if len(flagged) > 0:
    print(flagged[['employee', 'employee_email', 'total_negative_msgs', 'max_negatives_in_30d_window']].to_string(index=False))
else:
    print("  No employees meet the flight risk threshold.")

# Save flight risk results
flight_risk_df.to_csv('flight_risk_results.csv', index=False)

# Visualization
fig, ax = plt.subplots(figsize=(12, max(6, len(flagged) * 0.5 + 2)))

if len(flagged) > 0:
    top_risks = flagged.head(20)  # Show top 20
    bar_colors = ['#e74c3c' if r else '#f39c12' for r in top_risks['is_flight_risk']]
    ax.barh(range(len(top_risks)), top_risks['max_negatives_in_30d_window'].values, 
            color=bar_colors, edgecolor='white')
    ax.set_yticks(range(len(top_risks)))
    ax.set_yticklabels(top_risks['employee'].values)
    ax.set_xlabel('Max Negative Messages in Any 30-Day Window')
    ax.set_title('Flight Risk Employees\n(>=4 negative messages in any 30-day window)', 
                fontsize=14, fontweight='bold')
    ax.axvline(x=4, color='red', linestyle='--', linewidth=2, label='Threshold (4)')
    ax.legend()
    ax.invert_yaxis()
    for i, v in enumerate(top_risks['max_negatives_in_30d_window'].values):
        ax.text(v + 0.1, i, str(v), va='center', fontweight='bold')
else:
    ax.text(0.5, 0.5, 'No employees flagged as flight risk', 
            ha='center', va='center', fontsize=16, transform=ax.transAxes)
    ax.set_title('Flight Risk Employees', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{VIZ_DIR}/flight_risk.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: flight_risk.png")


# ============================================================================
# TASK 6: PREDICTIVE MODELING
# ============================================================================
print(f"\n{'='*60}")
print("TASK 6: Predictive Modeling (Linear Regression)")
print(f"{'='*60}")

print("""
MODELING APPROACH:
------------------
Objective: Predict an employee's monthly sentiment score using message 
           characteristics as features.

Target variable: Monthly sentiment score (sum of +1/0/-1 per employee-month)

Features:
  1. msg_count        -- Total messages sent that month
  2. avg_msg_length   -- Average message length (characters)
  3. avg_word_count   -- Average word count per message
  4. positive_ratio   -- Fraction of positive messages
  5. negative_ratio   -- Fraction of negative messages
  6. total_words      -- Total words written that month

Model: sklearn LinearRegression
Split: 80% train / 20% test
Metrics: R², MAE, RMSE
""")

# Build feature matrix from monthly aggregation
features_df = df.groupby(['employee', 'year_month']).agg(
    total_score=('score', 'sum'),
    msg_count=('score', 'count'),
    avg_msg_length=('msg_length', 'mean'),
    avg_word_count=('word_count', 'mean'),
    total_words=('word_count', 'sum'),
    positive_count=('score', lambda x: (x == 1).sum()),
    negative_count=('score', lambda x: (x == -1).sum()),
).reset_index()

features_df['positive_ratio'] = features_df['positive_count'] / features_df['msg_count']
features_df['negative_ratio'] = features_df['negative_count'] / features_df['msg_count']

# Define features and target
feature_cols = ['msg_count', 'avg_msg_length', 'avg_word_count', 'total_words', 
                'positive_ratio', 'negative_ratio']
target_col = 'total_score'

X = features_df[feature_cols].fillna(0)
y = features_df[target_col]

print(f"Feature matrix shape: {X.shape}")
print(f"Target variable stats: mean={y.mean():.2f}, std={y.std():.2f}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set:  {len(X_test)} samples")

# Train linear regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predictions
y_pred_train = lr_model.predict(X_train)
y_pred_test = lr_model.predict(X_test)

# Evaluation metrics
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"\n-- MODEL PERFORMANCE --")
print(f"  Training R²:  {r2_train:.4f}")
print(f"  Test R²:      {r2_test:.4f}")
print(f"  Test MAE:     {mae_test:.4f}")
print(f"  Test RMSE:    {rmse_test:.4f}")

# Feature importance (coefficients)
print(f"\n-- FEATURE COEFFICIENTS --")
coef_df = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': lr_model.coef_
}).sort_values('Coefficient', ascending=False)

print(coef_df.to_string(index=False))
print(f"\n  Intercept: {lr_model.intercept_:.4f}")

print("""
INTERPRETATION:
---------------""")
for _, row in coef_df.iterrows():
    direction = "positively" if row['Coefficient'] > 0 else "negatively"
    print(f"  * {row['Feature']}: A unit increase is associated with a "
          f"{abs(row['Coefficient']):.4f} change in sentiment score ({direction})")

# Visualization: Actual vs Predicted
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Actual vs Predicted scatter
axes[0].scatter(y_test, y_pred_test, alpha=0.5, color='#3498db', s=30)
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
axes[0].set_xlabel('Actual Score', fontsize=12)
axes[0].set_ylabel('Predicted Score', fontsize=12)
axes[0].set_title(f'Actual vs Predicted Sentiment Score\n(R² = {r2_test:.4f})', fontsize=13, fontweight='bold')
axes[0].legend()

# 2. Residual plot
residuals = y_test - y_pred_test
axes[1].scatter(y_pred_test, residuals, alpha=0.5, color='#e67e22', s=30)
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predicted Score', fontsize=12)
axes[1].set_ylabel('Residual', fontsize=12)
axes[1].set_title('Residual Plot', fontsize=13, fontweight='bold')

# 3. Feature importance
coef_sorted = coef_df.sort_values('Coefficient')
bar_colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in coef_sorted['Coefficient']]
axes[2].barh(coef_sorted['Feature'], coef_sorted['Coefficient'], color=bar_colors, edgecolor='white')
axes[2].set_xlabel('Coefficient Value', fontsize=12)
axes[2].set_title('Feature Importance\n(Linear Regression Coefficients)', fontsize=13, fontweight='bold')
axes[2].axvline(x=0, color='black', linewidth=0.8)

plt.tight_layout()
plt.savefig(f'{VIZ_DIR}/model_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n  Saved: model_performance.png")

# Residual distribution
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(residuals, bins=30, color='#9b59b6', alpha=0.7, edgecolor='white')
ax.set_xlabel('Residual Value')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Prediction Residuals', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax.axvline(x=residuals.mean(), color='blue', linestyle='--', label=f'Mean: {residuals.mean():.3f}')
ax.legend()
plt.tight_layout()
plt.savefig(f'{VIZ_DIR}/residual_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: residual_distribution.png")


# ============================================================================
# SUMMARY & KEY RESULTS
# ============================================================================
print(f"\n{'='*60}")
print("ANALYSIS COMPLETE -- SUMMARY")
print(f"{'='*60}")

# Overall top 3 positive
overall_top3_pos = overall_scores.head(3)
overall_top3_neg = overall_scores.tail(3).sort_values()

print(f"\n-- TOP 3 POSITIVE EMPLOYEES (Overall) --")
for i, (name, score) in enumerate(overall_top3_pos.items(), 1):
    print(f"  #{i} {name} -- Score: {score}")

print(f"\n-- TOP 3 NEGATIVE EMPLOYEES (Overall) --")
for i, (name, score) in enumerate(overall_top3_neg.items(), 1):
    print(f"  #{i} {name} -- Score: {score}")

print(f"\n-- FLIGHT RISK EMPLOYEES --")
if len(flagged) > 0:
    for _, row in flagged.iterrows():
        print(f"  [!] {row['employee']} ({row['employee_email']}) -- "
              f"{row['max_negatives_in_30d_window']} negatives in 30-day window")
else:
    print("  No employees flagged.")

print(f"\n-- MODEL PERFORMANCE --")
print(f"  R² = {r2_test:.4f}, MAE = {mae_test:.4f}, RMSE = {rmse_test:.4f}")

# Store results for README/report generation
results = {
    'top3_positive': overall_top3_pos,
    'top3_negative': overall_top3_neg,
    'flagged_employees': flagged,
    'r2': r2_test,
    'mae': mae_test,
    'rmse': rmse_test,
    'total_records': len(df),
    'total_employees': df['employee'].nunique(),
    'sentiment_counts': df['sentiment'].value_counts(),
    'coef_df': coef_df,
    'lr_intercept': lr_model.intercept_,
}

# Save results dict for report generation
import pickle
with open('analysis_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print(f"\nAll outputs saved. Results pickled to 'analysis_results.pkl'")
print(f"Visualizations saved to '{VIZ_DIR}/' directory")
print(f"\n{'='*60}")
print("PIPELINE COMPLETE")
print(f"{'='*60}")
