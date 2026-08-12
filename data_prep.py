# ============================================================================
# data_prep.py -- Data Loading, Cleaning & Preprocessing
# Employee Sentiment Analysis Project
# ============================================================================

import pandas as pd
import numpy as np
import re
import os
import quopri

def load_and_clean_data(filepath='test(in).csv'):
    """
    Load the raw CSV dataset and perform cleaning operations.
    
    Steps:
    1. Read CSV with proper handling of multiline fields
    2. Parse dates
    3. Clean MIME/quoted-printable encoding artifacts from body text
    4. Extract employee name from email address
    5. Handle missing values
    
    Returns:
        pd.DataFrame: Cleaned DataFrame ready for analysis
    """
    print("=" * 60)
    print("PHASE 0: Data Loading & Cleaning")
    print("=" * 60)
    
    # ------------------------------------------------------------------
    # Step 1: Load raw CSV
    # ------------------------------------------------------------------
    print("\n[1/5] Loading raw CSV...")
    df = pd.read_csv(filepath, encoding='utf-8')
    print(f"  Loaded {len(df)} records with columns: {list(df.columns)}")
    
    # ------------------------------------------------------------------
    # Step 2: Clean quoted-printable encoding artifacts in body
    # ------------------------------------------------------------------
    print("\n[2/5] Cleaning MIME/quoted-printable artifacts...")
    
    def clean_body(text):
        """Remove quoted-printable encoding artifacts and clean text."""
        if pd.isna(text):
            return ""
        text = str(text)
        # Try to decode quoted-printable encoding
        try:
            # Replace soft line breaks (=\n)
            text = text.replace('=\n', '')
            # Common QP artifacts
            text = text.replace('=20', ' ')
            text = text.replace('=09', '\t')
            text = text.replace('=01)', '-')
            text = text.replace('=01', '')
            text = text.replace('=02', '')
            text = text.replace('=3D', '=')
            text = text.replace('=2E', '.')
            text = text.replace('=2C', ',')
            # Remove remaining =XX patterns (hex encoded chars)
            text = re.sub(r'=[0-9A-Fa-f]{2}', '', text)
        except Exception:
            pass
        # Clean excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        return text
    
    df['body_clean'] = df['body'].apply(clean_body)
    
    # Also clean subject
    df['Subject'] = df['Subject'].fillna('(No Subject)')
    df['Subject'] = df['Subject'].apply(lambda x: clean_body(x) if isinstance(x, str) else x)
    
    # Combine subject and body for analysis
    df['full_text'] = df['Subject'] + '. ' + df['body_clean']
    
    print(f"  Cleaned {len(df)} message bodies")
    
    # ------------------------------------------------------------------
    # Step 3: Parse dates
    # ------------------------------------------------------------------
    print("\n[3/5] Parsing dates...")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Drop rows with unparseable dates
    date_nulls = df['date'].isna().sum()
    if date_nulls > 0:
        print(f"  WARNING: {date_nulls} rows have unparseable dates -- dropping them")
        df = df.dropna(subset=['date'])
    
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['year_month'] = df['date'].dt.to_period('M')
    df['day_of_week'] = df['date'].dt.day_name()
    
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    # ------------------------------------------------------------------
    # Step 4: Extract employee info from email
    # ------------------------------------------------------------------
    print("\n[4/5] Extracting employee names from email addresses...")
    
    df['from'] = df['from'].fillna('unknown@enron.com')
    
    def extract_employee_name(email):
        """Extract a readable name from an email address."""
        if pd.isna(email):
            return "Unknown"
        local_part = email.split('@')[0]
        # Replace dots and underscores with spaces, title case
        name = local_part.replace('.', ' ').replace('_', ' ').title()
        return name
    
    df['employee'] = df['from'].apply(extract_employee_name)
    df['employee_email'] = df['from'].str.lower().str.strip()
    
    print(f"  Found {df['employee'].nunique()} unique employees")
    
    # ------------------------------------------------------------------
    # Step 5: Handle missing values & final cleanup
    # ------------------------------------------------------------------
    print("\n[5/5] Final cleanup...")
    
    # Message length features (useful later)
    df['msg_length'] = df['body_clean'].str.len()
    df['word_count'] = df['body_clean'].str.split().str.len().fillna(0).astype(int)
    
    # Reset index
    df = df.reset_index(drop=True)
    
    print(f"  Final dataset shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    
    return df


def save_cleaned_data(df, output_path='cleaned_data.csv'):
    """Save the cleaned DataFrame to CSV for reproducibility."""
    df.to_csv(output_path, index=False)
    print(f"\n  Saved cleaned data to {output_path}")


if __name__ == '__main__':
    df = load_and_clean_data()
    save_cleaned_data(df)
    
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  Records: {len(df)}")
    print(f"  Employees: {df['employee'].nunique()}")
    print(f"  Date range: {df['date'].min().date()} -- {df['date'].max().date()}")
    print(f"  Missing body: {(df['body_clean'] == '').sum()}")
    print(f"  Avg message length: {df['msg_length'].mean():.0f} chars")
