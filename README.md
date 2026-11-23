# Employee Sentiment Analysis

> **Project:** Employee Sentiment & Engagement Analysis  
> **Dataset:** Enron Email Corpus (`test(in).csv`)  
> **Date:** August 12, 2026

---

## Project Overview

This project analyzes employee communication patterns to evaluate sentiment and engagement levels. Using Natural Language Processing (NLP) and machine learning techniques, we:

1. **Labeled** ~2,191 messages as Positive, Negative, or Neutral using Hugging Face's `cardiffnlp/twitter-roberta-base-sentiment-latest` transformer model
2. **Explored** communication and organizational dynamics through comprehensive EDA with 13 custom Matplotlib/Seaborn visualizations
3. **Scored** employees on a monthly basis using a standardized scoring rubric (+1 for Positive, 0 for Neutral, -1 for Negative)
4. **Ranked** employees by aggregated sentiment scores to highlight constructive leaders and communication bottlenecks
5. **Identified** flight-risk employees using a rolling 30-day temporal window detector (threshold: $\ge 4$ negative emails)
6. **Built** an OLS linear regression model ($R^2 = 0.5942$) predicting employee monthly sentiment based on communication volume, length, and temporal cadence

---

## Key Results

### 🏆 Top 3 Most Positive Employees (Overall)

| Rank | Employee | Total Score |
|------|----------|-------------|
| 1 | Johnny Palmer | 41 |
| 2 | Sally Beck | 40 |
| 3 | Lydia Delgado | 38 |

### ⚠️ Top 3 Most Negative Employees (Overall)

| Rank | Employee | Total Score |
|------|----------|-------------|
| 1 | John Arnold | 15 |
| 2 | Kayne Coulter | 16 |
| 3 | Rhonda Denton | 16 |

### 🚨 Flight Risk Employees

Employees who sent **4 or more negative messages within any rolling 30-day window**:

*No employees met the flight risk threshold.*

---

## Sentiment Distribution

| Sentiment | Count | Percentage |
|-----------|-------|------------|
| Neutral | 1,665 | 76.0% |
| Positive | 403 | 18.4% |
| Negative | 123 | 5.6% |

---

## Predictive Model Performance

| Metric | Value |
|--------|-------|
| R² (Test) | 0.5942 |
| MAE (Test) | 0.6262 |
| RMSE (Test) | 0.9369 |

---

## Key Insights & HR Action Plan

1. **Organizational Sentiment Baseline**:
   - **76.0% Neutral (1,665 messages)**, **18.4% Positive (403 messages)**, and **5.6% Negative (123 messages)**.
   - The low proportion of negative communications indicates that standard day-to-day operations remain constructive, with spikes of negative sentiment corresponding to urgent deadlines and restructuring events.

2. **Constructive Leadership & Positive Champions**:
   - Top positive employees (Johnny Palmer: +41, Sally Beck: +40, Lydia Delgado: +38) maintain high engagement and positive morale across long messaging tenures.
   - **HR Recommendation**: Leverage top constructive communicators as mentors and team leads during organizational transitions.

3. **Flight Risk Detection & Preventative Intervention**:
   - Evaluated using a rolling 30-day temporal window looking for $\ge 4$ negative messages from an individual contributor.
   - While no employee crossed the critical 4-negative threshold in this sample, employees with low or negative net scores (e.g., John Arnold: 15, Kayne Coulter: 16, Rhonda Denton: 16) represent individuals experiencing operational friction.
   - **HR Action Protocol**:
     - Schedule proactive 1-on-1 check-ins with managers before burnout escalates.
     - Review quarterly workload distribution and project staffing.
     - Offer confidential Employee Assistance Program (EAP) support.

4. **Predictive Modeling Findings ($R^2 = 0.5942$)**:
   - The linear regression model ($R^2 \approx 0.59$, MAE = 0.6262, RMSE = 0.9369) demonstrates that sentiment score can be reliably predicted using message cadence, word count, and sentiment density.
   - Message length and communicative frequency are the highest-weighted predictive features.

---

## Visualizations Catalog (13 Custom Matplotlib/Seaborn Charts)

All 13 visualizations are generated into the `visualization/` directory:

| # | Chart Name | Analytical Purpose |
|---|---|---|
| 1 | `sentiment_distribution.png` | Breakdown of Positive, Neutral, and Negative message volume |
| 2 | `sentiment_trends_over_time.png` | Weekly/monthly sentiment score timeline showing organizational trajectory |
| 3 | `top_senders.png` | Highest volume email senders across the corpus |
| 4 | `message_length_distribution.png` | Character and token distribution across all emails |
| 5 | `message_length_by_sentiment.png` | Boxplots comparing message length across sentiment classes |
| 6 | `sentiment_by_day_of_week.png` | Communication tone patterns from Monday to Sunday |
| 7 | `word_clouds.png` | Lexical frequency clouds for positive vs negative messages |
| 8 | `anomaly_detection.png` | Statistical outlier detection in daily email frequency |
| 9 | `employee_rankings.png` | Overall employee leaderboard by cumulative sentiment score |
| 10 | `monthly_ranking_latest.png` | Point-in-time rankings for the most recent operating month |
| 11 | `flight_risk.png` | Rolling 30-day window negative message distribution |
| 12 | `model_performance.png` | Actual vs. predicted sentiment scores for linear regression ($R^2 = 0.59$) |
| 13 | `residual_distribution.png` | Error residual distribution verifying model regression assumptions |

---

## Project Structure

```
├── test(in).csv                  # Raw Enron email dataset (~2,191 messages)
├── cleaned_data.csv              # Cleaned & preprocessed dataset
├── labeled_data.csv              # Data with RoBERTa sentiment classifications
├── monthly_scores.csv            # Aggregated monthly employee sentiment scores
├── rankings_positive.csv         # Leaderboard of top positive employees
├── rankings_negative.csv         # Leaderboard of lowest sentiment employees
├── flight_risk_results.csv       # 30-day rolling window flight risk audit
├── requirements.txt              # Pinned Python package dependencies
├── data_prep.py                  # MIME decoding, email extraction & cleaning
├── sentiment_analysis.py         # End-to-end NLP & predictive pipeline (Tasks 1-6)
├── generate_report.py            # Automated executive report (.docx) & README exporter
├── Employee_Sentiment_Analysis.ipynb  # Interactive narrative Jupyter notebook
├── Final_Report.docx             # Comprehensive corporate report
├── README.md                     # Project documentation
└── visualization/                # 13 high-resolution visualization charts
```

---

## How to Reproduce

```bash
# 1. Clone repository
git clone https://github.com/raunaknayak12/Employee-Sentiment-Analysis.git
cd Employee-Sentiment-Analysis

# 2. Install dependencies (Python 3.10 - 3.12 recommended)
pip install -r requirements.txt

# 3. Run the complete analysis pipeline
python sentiment_analysis.py

# 4. Generate report & documentation
python generate_report.py

# 5. (Optional) Run interactive Jupyter notebook
jupyter notebook Employee_Sentiment_Analysis.ipynb
```

---

## Technologies Used

- **Python 3.12**
- **PyTorch & Hugging Face Transformers**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **scikit-learn**: Linear Regression, train/test split, $R^2$/MAE/RMSE evaluation
- **pandas & NumPy**: Rolling 30-day windows, time-series aggregation, data cleaning
- **Matplotlib & Seaborn**: 13 publication-grade charts and statistical plots
- **python-docx**: Automated DOCX report compilation

---

## Author & Internship Context

- **Developer**: Raunak Nayak ([GitHub](https://github.com/raunaknayak12) · [LinkedIn](https://linkedin.com/in/raunak-nayak-5b937732a))
- **Context**: Artificial Intelligence Intern — Springer Capital Investments LLC (Aug 2026 – Nov 2026)
