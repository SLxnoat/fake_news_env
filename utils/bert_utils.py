
"""
BERT Model Utilities for Team Integration
Created by Member 3
"""
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import numpy as np

def load_bert_model(model_path='../models/saved_bert_model'):
    """Load saved BERT model and tokenizer"""
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        return model, tokenizer, device
    except Exception as e:
        print(f"Error loading BERT model: {e}")
        return None, None, None

def predict_with_bert_simple(text, model_path='../models/saved_bert_model'):
    """Simple prediction function for Streamlit app"""
    model, tokenizer, device = load_bert_model(model_path)
    
    if model is None:
        return None
    
    # Tokenize and predict
    inputs = tokenizer(text, return_tensors='pt', truncation=True, 
                      padding=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions.max().item()
    
    return {
        'prediction': predicted_class,
        'confidence': confidence,
        'probabilities': predictions[0].cpu().numpy()
    }

def get_bert_embeddings(texts, model_path='../models/saved_bert_model'):
    """Get BERT embeddings for hybrid model (Member 4)"""
    model, tokenizer, device = load_bert_model(model_path)
    
    if model is None:
        return None
    
    embeddings = []
    
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, 
                          padding=True, max_length=128)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Use the [CLS] token embedding from last hidden state
            embedding = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
            embeddings.append(embedding.squeeze())
    
    return np.array(embeddings)
