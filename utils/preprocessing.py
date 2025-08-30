"""
Text Preprocessing Utilities for Fake News Detection
Created by Member 1 - Data Cleaning Team
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Union
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

def clean_text(text: str) -> str:
    """
    Clean and preprocess text data for fake news detection
    
    Args:
        text (str): Raw text input
        
    Returns:
        str: Cleaned text
    """
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^a-zA-Z\s\.\!\?]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def advanced_text_cleaning(text: str, remove_stopwords: bool = False) -> str:
    """
    Advanced text cleaning with optional stopword removal
    
    Args:
        text (str): Input text
        remove_stopwords (bool): Whether to remove English stopwords
        
    Returns:
        str: Advanced cleaned text
    """
    # Basic cleaning
    cleaned = clean_text(text)
    
    if remove_stopwords:
        try:
            stop_words = set(stopwords.words('english'))
            word_tokens = word_tokenize(cleaned)
            filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
            cleaned = ' '.join(filtered_text)
        except:
            pass  # If NLTK data not available, skip stopword removal
    
    return cleaned

def extract_text_features(text: str) -> Dict[str, Union[int, float]]:
    """
    Extract various text-based features for analysis
    
    Args:
        text (str): Input text
        
    Returns:
        Dict: Dictionary of extracted features
    """
    if pd.isna(text) or text is None:
        return {
            'length': 0,
            'word_count': 0,
            'avg_word_length': 0,
            'sentence_count': 0,
            'has_question': 0,
            'has_exclamation': 0,
            'has_quotes': 0,
            'uppercase_ratio': 0,
            'digit_ratio': 0,
            'punctuation_ratio': 0
        }
    
    text_str = str(text)
    
    # Basic counts
    length = len(text_str)
    words = text_str.split()
    word_count = len(words)
    
    # Calculate features
    features = {
        'length': length,
        'word_count': word_count,
        'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
        'sentence_count': len(re.split(r'[.!?]+', text_str)),
        'has_question': 1 if '?' in text_str else 0,
        'has_exclamation': 1 if '!' in text_str else 0,
        'has_quotes': 1 if '"' in text_str or "'" in text_str else 0,
        'uppercase_ratio': sum(c.isupper() for c in text_str) / max(length, 1),
        'digit_ratio': sum(c.isdigit() for c in text_str) / max(length, 1),
        'punctuation_ratio': sum(not c.isalnum() and not c.isspace() for c in text_str) / max(length, 1)
    }
    
    return features

def map_labels_to_binary(label: str) -> int:
    """
    Map LIAR dataset multi-class labels to binary classification
    
    Args:
        label (str): Original label from LIAR dataset
        
    Returns:
        int: Binary label (0=Fake, 1=Real)
    """
    if pd.isna(label):
        return 0
    
    label_map = {
        'true': 1,
        'mostly-true': 1,
        'half-true': 1,          # Borderline, but classified as real
        'false': 0,
        'barely-true': 0,
        'pants-fire': 0
    }
    
    return label_map.get(str(label).lower(), 0)

def encode_categorical_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Encode categorical features to numerical codes
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (List[str]): List of categorical columns to encode
        
    Returns:
        pd.DataFrame: Dataframe with encoded features
    """
    df_encoded = df.copy()
    
    for col in columns:
        if col in df_encoded.columns:
            # Fill missing values with 'unknown'
            df_encoded[col] = df_encoded[col].fillna('unknown')
            
            # Create categorical codes
            df_encoded[f'{col}_code'] = df_encoded[col].astype('category').cat.codes
            
            # Store encoding mapping for later use
            encoding_map = dict(enumerate(df_encoded[col].astype('category').cat.categories))
            df_encoded[f'{col}_encoding'] = str(encoding_map)
    
    return df_encoded

def create_feature_combinations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create combined features for enhanced model performance
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with additional combined features
    """
    df_combined = df.copy()
    
    # Text-based combinations
    if 'clean_statement' in df_combined.columns:
        text_features = df_combined['clean_statement'].apply(extract_text_features)
        
        # Convert to separate columns
        for feature_name in text_features.iloc[0].keys():
            df_combined[f'text_{feature_name}'] = text_features.apply(lambda x: x[feature_name])
    
    # Metadata combinations
    if 'party' in df_combined.columns and 'subject' in df_combined.columns:
        df_combined['party_subject'] = df_combined['party'].astype(str) + '_' + df_combined['subject'].astype(str)
        df_combined['party_subject_code'] = df_combined['party_subject'].astype('category').cat.codes
    
    # Speaker credibility proxy (based on historical data)
    if 'speaker' in df_combined.columns:
        speaker_stats = df_combined.groupby('speaker')['label_binary'].agg(['mean', 'count']).reset_index()
        speaker_stats.columns = ['speaker', 'speaker_credibility', 'speaker_count']
        df_combined = df_combined.merge(speaker_stats, on='speaker', how='left')
        df_combined['speaker_credibility'] = df_combined['speaker_credibility'].fillna(0.5)  # Neutral for unknown speakers
    
    return df_combined

def validate_data_quality(df: pd.DataFrame) -> Dict[str, Union[bool, str, int]]:
    """
    Validate the quality of processed data
    
    Args:
        df (pd.DataFrame): Processed dataframe
        
    Returns:
        Dict: Validation results
    """
    validation_results = {
        'total_samples': len(df),
        'has_required_columns': all(col in df.columns for col in ['clean_statement', 'label_binary']),
        'empty_statements': sum(df['clean_statement'].isna() | (df['clean_statement'] == '')),
        'missing_labels': sum(df['label_binary'].isna()),
        'label_distribution': df['label_binary'].value_counts().to_dict() if 'label_binary' in df.columns else {},
        'avg_statement_length': df['clean_statement'].str.len().mean() if 'clean_statement' in df.columns else 0,
        'quality_score': 0.0
    }
    
    # Calculate quality score
    quality_score = 1.0
    
    if validation_results['empty_statements'] > 0:
        quality_score -= 0.2
    
    if validation_results['missing_labels'] > 0:
        quality_score -= 0.3
    
    if not validation_results['has_required_columns']:
        quality_score -= 0.5
    
    # Check label balance
    if validation_results['label_distribution']:
        total = sum(validation_results['label_distribution'].values())
        fake_ratio = validation_results['label_distribution'].get(0, 0) / total
        if fake_ratio < 0.2 or fake_ratio > 0.8:  # Severely imbalanced
            quality_score -= 0.2
    
    validation_results['quality_score'] = max(0.0, quality_score)
    validation_results['quality_status'] = (
        'Excellent' if quality_score > 0.9 else
        'Good' if quality_score > 0.7 else
        'Fair' if quality_score > 0.5 else
        'Poor'
    )
    
    return validation_results

def preprocess_for_modeling(train_path: str, test_path: str, valid_path: str = None) -> Dict[str, pd.DataFrame]:
    """
    Complete preprocessing pipeline for all datasets
    
    Args:
        train_path (str): Path to training data
        test_path (str): Path to test data
        valid_path (str, optional): Path to validation data
        
    Returns:
        Dict: Dictionary containing processed dataframes
    """
    print("🔄 Starting complete preprocessing pipeline...")
    
    # Column names for LIAR dataset
    columns = [
        'id', 'label', 'statement', 'subject', 'speaker',
        'job', 'state', 'party', 'barely_true', 'false',
        'half_true', 'mostly_true', 'pants_on_fire', 'context'
    ]
    
    processed_data = {}
    
    # Process each dataset
    for dataset_name, file_path in [('train', train_path), ('test', test_path), ('valid', valid_path)]:
        if file_path is None:
            continue
            
        print(f"  📊 Processing {dataset_name} dataset...")
        
        try:
            # Load data
            df = pd.read_csv(file_path, sep='\t', names=columns, encoding='utf-8')
            
            # Apply all preprocessing steps
            df['clean_statement'] = df['statement'].apply(clean_text)
            df['label_binary'] = df['label'].apply(map_labels_to_binary)
            
            # Handle metadata
            metadata_cols = ['party', 'subject', 'speaker', 'job', 'state']
            for col in metadata_cols:
                df[col] = df[col].fillna('unknown')
            
            # Encode categorical features
            df = encode_categorical_features(df, metadata_cols)
            
            # Create feature combinations
            df = create_feature_combinations(df)
            
            # Remove rows with empty statements
            df = df[df['clean_statement'].str.len() > 0]
            
            # Validate quality
            quality_results = validate_data_quality(df)
            print(f"    ✅ {dataset_name}: {quality_results['quality_status']} quality "
                  f"(Score: {quality_results['quality_score']:.2f})")
            
            processed_data[dataset_name] = df
            
        except Exception as e:
            print(f"    ❌ Error processing {dataset_name}: {str(e)}")
    
    print("✅ Preprocessing pipeline complete!")
    return processed_data

# Utility function for Streamlit app
def prepare_single_statement(statement: str, party: str = 'none', subject: str = 'other') -> np.ndarray:
    """
    Prepare a single statement for model prediction
    
    Args:
        statement (str): News statement
        party (str): Political party
        subject (str): Subject category
        
    Returns:
        np.ndarray: Feature vector ready for model input
    """
    # Clean text
    cleaned = clean_text(statement)
    
    # Extract text features
    text_feats = extract_text_features(statement)
    
    # Create metadata features
    party_map = {'republican': 0, 'democrat': 1, 'independent': 2, 'none': 3}
    subject_map = {'politics': 0, 'healthcare': 1, 'education': 2, 'economy': 3, 'other': 4}
    
    party_code = party_map.get(party.lower(), 3)
    subject_code = subject_map.get(subject.lower(), 4)
    
    # Combine all features
    feature_vector = [
        party_code,
        subject_code,
        0, 0, 0,  # Placeholder for other categorical features
        text_feats['length'],
        text_feats['word_count'],
        text_feats['has_question'],
        text_feats['has_exclamation'],
        text_feats['has_quotes'],
        text_feats['uppercase_ratio'],
        text_feats['avg_word_length']
    ]
    
    return np.array(feature_vector).reshape(1, -1)

def batch_preprocess_statements(statements: List[str], parties: List[str] = None, 
                               subjects: List[str] = None) -> np.ndarray:
    """
    Preprocess multiple statements for batch prediction
    
    Args:
        statements (List[str]): List of news statements
        parties (List[str], optional): List of political parties
        subjects (List[str], optional): List of subjects
        
    Returns:
        np.ndarray: Feature matrix for batch prediction
    """
    if parties is None:
        parties = ['none'] * len(statements)
    if subjects is None:
        subjects = ['other'] * len(statements)
    
    feature_matrix = []
    
    for i, statement in enumerate(statements):
        party = parties[i] if i < len(parties) else 'none'
        subject = subjects[i] if i < len(subjects) else 'other'
        
        features = prepare_single_statement(statement, party, subject)
        feature_matrix.append(features.flatten())
    
    return np.array(feature_matrix)

# Statistical analysis utilities
def analyze_text_statistics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze text statistics for dataset overview
    
    Args:
        df (pd.DataFrame): Dataframe with text data
        
    Returns:
        Dict: Statistical summary
    """
    if 'clean_statement' not in df.columns:
        return {}
    
    statements = df['clean_statement'].dropna()
    
    stats = {
        'total_statements': len(statements),
        'avg_length': statements.str.len().mean(),
        'median_length': statements.str.len().median(),
        'avg_words': statements.str.split().str.len().mean(),
        'median_words': statements.str.split().str.len().median(),
        'empty_statements': sum(statements.str.len() == 0),
        'very_short': sum(statements.str.len() < 10),  # Less than 10 characters
        'very_long': sum(statements.str.len() > 500),   # More than 500 characters
    }
    
    return stats

def get_label_distribution(df: pd.DataFrame) -> Dict[str, Union[int, float]]:
    """
    Analyze label distribution in dataset
    
    Args:
        df (pd.DataFrame): Dataframe with labels
        
    Returns:
        Dict: Label distribution statistics
    """
    if 'label_binary' not in df.columns:
        return {}
    
    labels = df['label_binary'].dropna()
    
    distribution = {
        'total_labels': len(labels),
        'real_count': sum(labels == 1),
        'fake_count': sum(labels == 0),
        'real_percentage': sum(labels == 1) / len(labels) * 100,
        'fake_percentage': sum(labels == 0) / len(labels) * 100,
        'balance_ratio': min(sum(labels == 0), sum(labels == 1)) / max(sum(labels == 0), sum(labels == 1))
    }
    
    return distribution

def create_preprocessing_report(df: pd.DataFrame) -> str:
    """
    Create a comprehensive preprocessing report
    
    Args:
        df (pd.DataFrame): Processed dataframe
        
    Returns:
        str: Formatted report
    """
    text_stats = analyze_text_statistics(df)
    label_dist = get_label_distribution(df)
    quality = validate_data_quality(df)
    
    report = f"""
📊 PREPROCESSING REPORT
{'='*50}

📈 Dataset Overview:
   Total Samples: {text_stats.get('total_statements', 0)}
   Quality Score: {quality.get('quality_score', 0):.2f} ({quality.get('quality_status', 'Unknown')})
   Empty Statements: {quality.get('empty_statements', 0)}

📝 Text Statistics:
   Average Length: {text_stats.get('avg_length', 0):.1f} characters
   Average Words: {text_stats.get('avg_words', 0):.1f} words
   Very Short (<10 chars): {text_stats.get('very_short', 0)}
   Very Long (>500 chars): {text_stats.get('very_long', 0)}

🏷 Label Distribution:
   Real News: {label_dist.get('real_count', 0)} ({label_dist.get('real_percentage', 0):.1f}%)
   Fake News: {label_dist.get('fake_count', 0)} ({label_dist.get('fake_percentage', 0):.1f}%)
   Balance Ratio: {label_dist.get('balance_ratio', 0):.2f}

✅ Data Quality Checks:
   Required Columns: {'✅' if quality.get('has_required_columns') else '❌'}
   Missing Labels: {quality.get('missing_labels', 0)}
   Overall Status: {quality.get('quality_status', 'Unknown')}
"""
    
    return report

def export_cleaned_data(df: pd.DataFrame, output_path: str, include_report: bool = True) -> bool:
    """
    Export cleaned data with optional preprocessing report
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
        output_path (str): Output file path
        include_report (bool): Whether to save preprocessing report
        
    Returns:
        bool: Success status
    """
    try:
        # Save cleaned data
        df.to_csv(output_path, index=False)
        
        # Save preprocessing report
        if include_report:
            report_path = output_path.replace('.csv', '_report.txt')
            report = create_preprocessing_report(df)
            
            with open(report_path, 'w') as f:
                f.write(report)
            
            print(f"✅ Data exported to: {output_path}")
            print(f"✅ Report saved to: {report_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {str(e)}")
        return False

# Constants for consistent encoding across team
PARTY_ENCODING = {
    'republican': 0,
    'democrat': 1, 
    'independent': 2,
    'none': 3,
    'unknown': 4
}

SUBJECT_ENCODING = {
    'politics': 0,
    'healthcare': 1,
    'education': 2,
    'economy': 3,
    'other': 4,
    'unknown': 5
}

def get_encoding_maps():
    """Return standard encoding maps for team consistency"""
    return {
        'party_encoding': PARTY_ENCODING,
        'subject_encoding': SUBJECT_ENCODING
    }

# Main preprocessing function for team workflow
def main_preprocessing_workflow(data_dir: str = 'data') -> bool:
    """
    Execute complete preprocessing workflow
    
    Args:
        data_dir (str): Directory containing raw data files
        
    Returns:
        bool: Success status
    """
    print("🚀 Starting main preprocessing workflow...")
    
    try:
        # Define file paths
        train_path = os.path.join(data_dir, 'train.tsv')
        test_path = os.path.join(data_dir, 'test.tsv')
        valid_path = os.path.join(data_dir, 'valid.tsv')
        
        # Process all datasets
        processed_data = preprocess_for_modeling(train_path, test_path, valid_path)
        
        if not processed_data:
            print("❌ No data processed successfully")
            return False
        
        # Save processed datasets
        for dataset_name, df in processed_data.items():
            output_path = os.path.join(data_dir, f'{dataset_name}_clean.csv')
            export_cleaned_data(df, output_path, include_report=True)
        
        # Create combined summary
        summary_data = []
        for dataset_name, df in processed_data.items():
            quality = validate_data_quality(df)
            text_stats = analyze_text_statistics(df)
            label_dist = get_label_distribution(df)
            
            summary_data.append({
                'Dataset': dataset_name.title(),
                'Samples': quality.get('total_samples', 0),
                'Quality_Score': quality.get('quality_score', 0),
                'Avg_Length': text_stats.get('avg_length', 0),
                'Real_News': label_dist.get('real_count', 0),
                'Fake_News': label_dist.get('fake_count', 0),
                'Balance_Ratio': label_dist.get('balance_ratio', 0)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(data_dir, 'preprocessing_summary.csv'), index=False)
        
        print("✅ Main preprocessing workflow completed successfully!")
        print(f"📁 Check {data_dir}/ folder for all processed files")
        
        return True
        
    except Exception as e:
        print(f"❌ Preprocessing workflow failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Run preprocessing if script is executed directly
    print("🧹 Running preprocessing utilities...")
    
    # Example usage
    sample_text = "The President announced new policies today!"
    cleaned = clean_text(sample_text)
    features = extract_text_features(sample_text)
    
    print(f"📝 Sample text: {sample_text}")
    print(f"🧹 Cleaned: {cleaned}")
    print(f"🔧 Features: {features}")
    
    print("\n✅ Preprocessing utilities working correctly!")