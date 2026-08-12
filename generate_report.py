# ============================================================================
# generate_report.py -- Generate Final Report (.docx) and README.md
# Employee Sentiment Analysis Project
# ============================================================================

import pickle
import pandas as pd
import os
from datetime import datetime

def generate_readme(results):
    """Generate the README.md file with key findings."""
    
    top3_pos = results['top3_positive']
    top3_neg = results['top3_negative']
    flagged = results['flagged_employees']
    sentiment_counts = results['sentiment_counts']
    
    readme = f"""# Employee Sentiment Analysis

> **Project:** Employee Sentiment & Engagement Analysis  
> **Dataset:** Enron Email Corpus (`test(in).csv`)  
> **Date:** {datetime.now().strftime('%B %d, %Y')}

---

## Project Overview

This project analyzes employee communication patterns to evaluate sentiment and engagement levels. Using Natural Language Processing (NLP) and machine learning techniques, we:

1. **Labeled** ~{results['total_records']:,} messages as Positive, Negative, or Neutral using a pre-trained RoBERTa transformer model
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
"""
    
    for i, (name, score) in enumerate(top3_pos.items(), 1):
        readme += f"| {i} | {name} | {score} |\n"
    
    readme += f"""
### [!]️ Top 3 Most Negative Employees (Overall)

| Rank | Employee | Total Score |
|------|----------|-------------|
"""
    
    for i, (name, score) in enumerate(top3_neg.items(), 1):
        readme += f"| {i} | {name} | {score} |\n"
    
    readme += f"""
### 🚨 Flight Risk Employees

Employees who sent **4 or more negative messages within any rolling 30-day window**:

"""
    
    if len(flagged) > 0:
        readme += "| Employee | Email | Negative Msgs (Total) | Max in 30-Day Window |\n"
        readme += "|----------|-------|-----------------------|----------------------|\n"
        for _, row in flagged.iterrows():
            readme += f"| {row['employee']} | {row['employee_email']} | {row['total_negative_msgs']} | {row['max_negatives_in_30d_window']} |\n"
    else:
        readme += "*No employees met the flight risk threshold.*\n"
    
    readme += f"""
---

## Sentiment Distribution

| Sentiment | Count | Percentage |
|-----------|-------|------------|
"""
    total = sentiment_counts.sum()
    for sent, count in sentiment_counts.items():
        readme += f"| {sent} | {count:,} | {count/total*100:.1f}% |\n"
    
    readme += f"""
---

## Predictive Model Performance

| Metric | Value |
|--------|-------|
| R² (Test) | {results['r2']:.4f} |
| MAE (Test) | {results['mae']:.4f} |
| RMSE (Test) | {results['rmse']:.4f} |

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
"""
    
    return readme


def generate_docx_report(results):
    """Generate the Final Report as a .docx file."""
    from docx import Document
    from docx.shared import Inches, Pt, Cm, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    
    doc = Document()
    
    # -- Title --
    title = doc.add_heading('Employee Sentiment Analysis -- Final Report', level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_paragraph(f'Date: {datetime.now().strftime("%B %d, %Y")}')
    doc.add_paragraph(f'Dataset: test(in).csv ({results["total_records"]:,} records, {results["total_employees"]} employees)')
    doc.add_paragraph('')
    
    # -- Section 1: Approach & Methodology --
    doc.add_heading('1. Approach & Methodology', level=1)
    
    doc.add_heading('1.1 Data Preparation', level=2)
    doc.add_paragraph(
        'The raw dataset was loaded from test(in).csv containing email communications. '
        'Data cleaning involved: removing MIME/quoted-printable encoding artifacts, '
        'parsing dates, extracting employee names from email addresses, and computing '
        'message length features. The cleaned dataset was saved for reproducibility.'
    )
    
    doc.add_heading('1.2 Sentiment Classification', level=2)
    doc.add_paragraph(
        'Sentiment labeling was performed using the cardiffnlp/twitter-roberta-base-sentiment-latest '
        'model from HuggingFace Transformers. This is a RoBERTa-based model fine-tuned on approximately '
        '124 million tweets for 3-class sentiment classification (Positive, Neutral, Negative). '
        'The model was chosen for its native 3-class output, strong validation performance, '
        'ability to handle informal text, and full local reproducibility without API dependencies. '
        'Each message\'s combined subject and body text was processed through the model pipeline, '
        'with texts truncated to 500 characters for computational efficiency.'
    )
    
    doc.add_heading('1.3 Scoring System', level=2)
    doc.add_paragraph(
        'Employee sentiment scores were calculated using a simple numeric mapping: '
        'Positive → +1, Neutral → 0, Negative → -1. Scores were aggregated per employee '
        'per calendar month, with scores resetting at the start of each new month.'
    )
    
    doc.add_heading('1.4 Flight Risk Identification', level=2)
    doc.add_paragraph(
        'Flight risk was defined as any employee who sent 4 or more negative messages '
        'within any rolling 30-day window. A sliding window algorithm was implemented: '
        'for each employee, all negative message dates were sorted, and for each date, '
        'a count of negative messages within the next 30 days was computed. This approach '
        'correctly handles overlapping windows and is not limited to calendar month boundaries.'
    )
    
    doc.add_heading('1.5 Predictive Modeling', level=2)
    doc.add_paragraph(
        'A linear regression model was trained to predict monthly employee sentiment scores '
        'using features extracted from message characteristics: message count, average message '
        'length, average word count, total words, positive message ratio, and negative message '
        'ratio. The data was split 80/20 for training and testing.'
    )
    
    # -- Section 2: Key EDA Findings --
    doc.add_heading('2. Key EDA Findings', level=1)
    
    doc.add_heading('2.1 Dataset Overview', level=2)
    doc.add_paragraph(f'Total records: {results["total_records"]:,}')
    doc.add_paragraph(f'Unique employees: {results["total_employees"]}')
    
    doc.add_heading('2.2 Sentiment Distribution', level=2)
    total = results['sentiment_counts'].sum()
    for sent, count in results['sentiment_counts'].items():
        doc.add_paragraph(f'{sent}: {count:,} ({count/total*100:.1f}%)')
    
    # Add visualization if available
    viz_path = 'visualization/sentiment_distribution.png'
    if os.path.exists(viz_path):
        doc.add_picture(viz_path, width=Inches(5.5))
        last_paragraph = doc.paragraphs[-1]
        last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_heading('2.3 Temporal Trends', level=2)
    doc.add_paragraph(
        'Monthly sentiment trends were analyzed to identify patterns over time. '
        'The analysis reveals periods of increased negative sentiment that may correlate '
        'with organizational events or external factors.'
    )
    
    viz_path = 'visualization/sentiment_trends_over_time.png'
    if os.path.exists(viz_path):
        doc.add_picture(viz_path, width=Inches(5.5))
        last_paragraph = doc.paragraphs[-1]
        last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_heading('2.4 Message Patterns', level=2)
    doc.add_paragraph(
        'Analysis of message length, word count, and day-of-week patterns provides '
        'additional context for understanding employee communication behaviors. '
        'Word clouds for each sentiment category reveal the most common terms '
        'associated with positive, neutral, and negative communications.'
    )
    
    # -- Section 3: Employee Scoring & Ranking --
    doc.add_heading('3. Employee Scoring & Ranking', level=1)
    
    doc.add_heading('3.1 Scoring Method', level=2)
    doc.add_paragraph(
        'Scores were computed by mapping each sentiment label to a numeric value '
        '(Positive: +1, Neutral: 0, Negative: -1) and summing per employee per month. '
        'Scores reset at the start of each calendar month.'
    )
    
    doc.add_heading('3.2 Top Positive Employees (Overall)', level=2)
    table = doc.add_table(rows=1, cols=3)
    table.style = 'Light Grid Accent 1'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Rank'
    hdr_cells[1].text = 'Employee'
    hdr_cells[2].text = 'Total Score'
    
    for i, (name, score) in enumerate(results['top3_positive'].items(), 1):
        row_cells = table.add_row().cells
        row_cells[0].text = str(i)
        row_cells[1].text = str(name)
        row_cells[2].text = str(score)
    
    doc.add_paragraph('')
    
    doc.add_heading('3.3 Top Negative Employees (Overall)', level=2)
    table = doc.add_table(rows=1, cols=3)
    table.style = 'Light Grid Accent 1'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Rank'
    hdr_cells[1].text = 'Employee'
    hdr_cells[2].text = 'Total Score'
    
    for i, (name, score) in enumerate(results['top3_negative'].items(), 1):
        row_cells = table.add_row().cells
        row_cells[0].text = str(i)
        row_cells[1].text = str(name)
        row_cells[2].text = str(score)
    
    viz_path = 'visualization/employee_rankings.png'
    if os.path.exists(viz_path):
        doc.add_paragraph('')
        doc.add_picture(viz_path, width=Inches(5.5))
        last_paragraph = doc.paragraphs[-1]
        last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # -- Section 4: Flight Risk --
    doc.add_heading('4. Flight Risk Criteria & Outcomes', level=1)
    
    doc.add_heading('4.1 Criteria', level=2)
    doc.add_paragraph(
        'An employee is flagged as a flight risk if they sent 4 or more negative messages '
        'within any rolling 30-day window. This is implemented using a sliding window algorithm '
        'that examines all possible 30-day windows, not just calendar months.'
    )
    
    doc.add_heading('4.2 Results', level=2)
    flagged = results['flagged_employees']
    doc.add_paragraph(f'Total employees flagged: {len(flagged)}')
    
    if len(flagged) > 0:
        table = doc.add_table(rows=1, cols=4)
        table.style = 'Light Grid Accent 1'
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = 'Employee'
        hdr_cells[1].text = 'Email'
        hdr_cells[2].text = 'Total Negative'
        hdr_cells[3].text = 'Max in 30-Day Window'
        
        for _, row in flagged.iterrows():
            row_cells = table.add_row().cells
            row_cells[0].text = str(row['employee'])
            row_cells[1].text = str(row['employee_email'])
            row_cells[2].text = str(row['total_negative_msgs'])
            row_cells[3].text = str(row['max_negatives_in_30d_window'])
    
    viz_path = 'visualization/flight_risk.png'
    if os.path.exists(viz_path):
        doc.add_paragraph('')
        doc.add_picture(viz_path, width=Inches(5.5))
        last_paragraph = doc.paragraphs[-1]
        last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # -- Section 5: Predictive Model --
    doc.add_heading('5. Predictive Model Overview & Evaluation', level=1)
    
    doc.add_heading('5.1 Model Details', level=2)
    doc.add_paragraph(
        'A linear regression model was trained to predict monthly employee sentiment scores. '
        'Features include message count, average message length, average word count, '
        'total words written, positive message ratio, and negative message ratio.'
    )
    
    doc.add_heading('5.2 Performance Metrics', level=2)
    table = doc.add_table(rows=1, cols=2)
    table.style = 'Light Grid Accent 1'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Metric'
    hdr_cells[1].text = 'Value'
    
    for metric, value in [('R² (Test)', f"{results['r2']:.4f}"), 
                           ('MAE (Test)', f"{results['mae']:.4f}"),
                           ('RMSE (Test)', f"{results['rmse']:.4f}")]:
        row_cells = table.add_row().cells
        row_cells[0].text = metric
        row_cells[1].text = value
    
    doc.add_heading('5.3 Feature Coefficients', level=2)
    coef_df = results['coef_df']
    table = doc.add_table(rows=1, cols=2)
    table.style = 'Light Grid Accent 1'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Feature'
    hdr_cells[1].text = 'Coefficient'
    
    for _, row in coef_df.iterrows():
        row_cells = table.add_row().cells
        row_cells[0].text = str(row['Feature'])
        row_cells[1].text = f"{row['Coefficient']:.4f}"
    
    row_cells = table.add_row().cells
    row_cells[0].text = 'Intercept'
    row_cells[1].text = f"{results['lr_intercept']:.4f}"
    
    doc.add_heading('5.4 Interpretation', level=2)
    doc.add_paragraph(
        'The linear regression model provides interpretable coefficients that reveal '
        'which message characteristics are most predictive of sentiment scores. '
        'The positive and negative message ratios are expected to have the strongest '
        'coefficients, as they directly reflect the sentiment composition. Other features '
        'like message count and length provide additional predictive power by capturing '
        'communication intensity and engagement patterns.'
    )
    
    viz_path = 'visualization/model_performance.png'
    if os.path.exists(viz_path):
        doc.add_paragraph('')
        doc.add_picture(viz_path, width=Inches(5.5))
        last_paragraph = doc.paragraphs[-1]
        last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # -- Section 6: Conclusions --
    doc.add_heading('6. Conclusions & Recommendations', level=1)
    doc.add_paragraph(
        'This analysis demonstrates a comprehensive approach to employee sentiment analysis '
        'using NLP and machine learning. Key recommendations include:'
    )
    doc.add_paragraph('Monitor flight-risk employees with proactive engagement strategies', style='List Bullet')
    doc.add_paragraph('Use sentiment trends to identify organizational events impacting morale', style='List Bullet')
    doc.add_paragraph('Leverage top positive employees as communication role models', style='List Bullet')
    doc.add_paragraph('Continuously update the predictive model with new data for improved accuracy', style='List Bullet')
    doc.add_paragraph('Consider additional features (e.g., response time, thread length) for future model improvements', style='List Bullet')
    
    # Save document
    doc.save('Final_Report.docx')
    print("Final_Report.docx saved successfully!")


if __name__ == '__main__':
    print("=" * 60)
    print("GENERATING DELIVERABLES")
    print("=" * 60)
    
    # Load results
    with open('analysis_results.pkl', 'rb') as f:
        results = pickle.load(f)
    
    # Generate README
    print("\n[1/2] Generating README.md...")
    readme_content = generate_readme(results)
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print("  README.md saved!")
    
    # Generate DOCX report
    print("\n[2/2] Generating Final_Report.docx...")
    generate_docx_report(results)
    
    print("\n" + "=" * 60)
    print("ALL DELIVERABLES GENERATED")
    print("=" * 60)
