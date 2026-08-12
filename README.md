# Employee Sentiment Analysis

> **Project:** Employee Sentiment & Engagement Analysis  
> **Dataset:** Enron Email Corpus (`test(in).csv`)  
> **Date:** August 12, 2026

---

## Project Overview

This project analyzes employee communication patterns to evaluate sentiment and engagement levels. Using Natural Language Processing (NLP) and machine learning techniques, we:

1. **Labeled** ~2,191 messages as Positive, Negative, or Neutral using a pre-trained RoBERTa transformer model
2. **Explored** data trends through comprehensive EDA with visualizations
3. **Scored** employees monthly based on sentiment (+1/0/-1)
4. **Ranked** employees by sentiment scores
5. **Identified** flight-risk employees using a rolling 30-day window analysis
6. **Built** a linear regression model to predict sentiment trends

---

## Key Results

### 🏆 Top 3 Most Positive Employees (Overall)

| Rank | Employee | Total Score |
|------|----------|-------------|
| 1 | Johnny Palmer | 41 |
| 2 | Sally Beck | 40 |
| 3 | Lydia Delgado | 38 |

### [!]️ Top 3 Most Negative Employees (Overall)

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

## Key Insights & Recommendations

1. **Sentiment Balance:** The dataset shows the distribution of sentiment across employee communications, revealing overall organizational tone.

2. **High-Engagement Employees:** The top-ranked positive employees consistently demonstrate constructive communication patterns. These employees could serve as models for organizational communication standards.

3. **At-Risk Employees:** Flight-risk employees identified through sustained negative messaging patterns should be prioritized for:
   - One-on-one check-ins with management
   - Employee assistance program (EAP) referrals
   - Workload and role satisfaction assessment

4. **Predictive Insights:** The linear regression model reveals which message characteristics most strongly correlate with sentiment scores, enabling proactive monitoring.

5. **Temporal Patterns:** Analysis of sentiment trends over time can help identify organizational events or policy changes that impact employee morale.

---

## Project Structure

```
├-- test(in).csv                  # Raw dataset
├-- cleaned_data.csv              # Cleaned & preprocessed data
├-- labeled_data.csv              # Data with sentiment labels
├-- monthly_scores.csv            # Monthly employee sentiment scores
├-- rankings_positive.csv         # Monthly top positive rankings
├-- rankings_negative.csv         # Monthly top negative rankings
├-- flight_risk_results.csv       # Flight risk analysis results
├-- data_prep.py                  # Data loading & cleaning module
├-- sentiment_analysis.py         # Main analysis pipeline (Tasks 1-6)
├-- generate_report.py            # Report & README generation
├-- Employee_Sentiment_Analysis.ipynb  # Jupyter notebook (full narrative)
├-- Final_Report.docx             # Final project report
├-- README.md                     # This file
└-- visualization/                # All charts and graphs
    ├-- sentiment_distribution.png
    ├-- sentiment_trends_over_time.png
    ├-- top_senders.png
    ├-- message_length_distribution.png
    ├-- message_length_by_sentiment.png
    ├-- sentiment_by_day_of_week.png
    ├-- word_clouds.png
    ├-- anomaly_detection.png
    ├-- employee_rankings.png
    ├-- monthly_ranking_latest.png
    ├-- flight_risk.png
    ├-- model_performance.png
    └-- residual_distribution.png
```

---

## How to Reproduce

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn torch transformers wordcloud openpyxl python-docx jupyter

# 2. Run the full pipeline
python sentiment_analysis.py

# 3. Generate report & README
python generate_report.py

# 4. (Optional) View as Jupyter notebook
jupyter notebook Employee_Sentiment_Analysis.ipynb
```

---

## Technologies Used

- **Python 3.13**
- **PyTorch + HuggingFace Transformers** -- Sentiment classification (RoBERTa)
- **scikit-learn** -- Linear regression modeling
- **pandas / NumPy** -- Data manipulation
- **matplotlib / seaborn** -- Visualization
- **python-docx** -- Report generation
