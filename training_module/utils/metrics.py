"""
Evaluation metrics for conversational AI models.
"""

import math
import re
from typing import List, Dict, Any
from collections import Counter

import numpy as np


def calculate_bleu_score(predictions: List[str], references: List[str], n_gram: int = 4) -> float:
    """
    Calculate BLEU score for generated responses.
    
    Args:
        predictions: List of generated responses
        references: List of reference responses
        n_gram: Maximum n-gram order
        
    Returns:
        BLEU score
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")
    
    bleu_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        
        if not pred_tokens:
            bleu_scores.append(0.0)
            continue
        
        # Calculate precision for each n-gram order
        precisions = []
        
        for n in range(1, min(n_gram + 1, len(pred_tokens) + 1)):
            pred_ngrams = get_ngrams(pred_tokens, n)
            ref_ngrams = get_ngrams(ref_tokens, n)
            
            if not pred_ngrams:
                precisions.append(0.0)
                continue
            
            matches = 0
            for ngram in pred_ngrams:
                if ngram in ref_ngrams:
                    matches += min(pred_ngrams[ngram], ref_ngrams[ngram])
            
            precision = matches / sum(pred_ngrams.values())
            precisions.append(precision)
        
        if not precisions or all(p == 0 for p in precisions):
            bleu_scores.append(0.0)
            continue
        
        # Calculate brevity penalty
        bp = brevity_penalty(len(pred_tokens), len(ref_tokens))
        
        # Calculate geometric mean of precisions
        log_precisions = [math.log(p) if p > 0 else float('-inf') for p in precisions]
        if any(p == float('-inf') for p in log_precisions):
            bleu_scores.append(0.0)
        else:
            bleu = bp * math.exp(sum(log_precisions) / len(log_precisions))
            bleu_scores.append(bleu)
    
    return sum(bleu_scores) / len(bleu_scores)


def get_ngrams(tokens: List[str], n: int) -> Counter:
    """Get n-grams from token list."""
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngrams.append(ngram)
    return Counter(ngrams)


def brevity_penalty(pred_len: int, ref_len: int) -> float:
    """Calculate brevity penalty for BLEU."""
    if pred_len > ref_len:
        return 1.0
    elif pred_len == 0:
        return 0.0
    else:
        return math.exp(1 - ref_len / pred_len)


def calculate_rouge_l(predictions: List[str], references: List[str]) -> float:
    """
    Calculate ROUGE-L score (Longest Common Subsequence).
    
    Args:
        predictions: List of generated responses
        references: List of reference responses
        
    Returns:
        ROUGE-L F1 score
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")
    
    rouge_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        
        if not pred_tokens or not ref_tokens:
            rouge_scores.append(0.0)
            continue
        
        # Calculate LCS
        lcs_length = longest_common_subsequence(pred_tokens, ref_tokens)
        
        # Calculate precision and recall
        precision = lcs_length / len(pred_tokens) if pred_tokens else 0
        recall = lcs_length / len(ref_tokens) if ref_tokens else 0
        
        # Calculate F1 score
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        rouge_scores.append(f1)
    
    return sum(rouge_scores) / len(rouge_scores)


def longest_common_subsequence(seq1: List[str], seq2: List[str]) -> int:
    """Calculate length of longest common subsequence."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]


def calculate_distinct_ngrams(responses: List[str], n: int = 2) -> float:
    """
    Calculate distinct n-gram ratio for diversity measurement.
    
    Args:
        responses: List of generated responses
        n: N-gram order
        
    Returns:
        Distinct n-gram ratio
    """
    all_ngrams = []
    
    for response in responses:
        tokens = response.lower().split()
        ngrams = get_ngrams(tokens, n)
        all_ngrams.extend(ngrams.elements())
    
    if not all_ngrams:
        return 0.0
    
    unique_ngrams = set(all_ngrams)
    return len(unique_ngrams) / len(all_ngrams)


def calculate_response_length_stats(responses: List[str]) -> Dict[str, float]:
    """
    Calculate response length statistics.
    
    Args:
        responses: List of generated responses
        
    Returns:
        Dictionary with length statistics
    """
    lengths = [len(response.split()) for response in responses]
    
    if not lengths:
        return {
            "avg_length": 0.0,
            "min_length": 0.0,
            "max_length": 0.0,
            "std_length": 0.0
        }
    
    return {
        "avg_length": np.mean(lengths),
        "min_length": float(min(lengths)),
        "max_length": float(max(lengths)),
        "std_length": np.std(lengths)
    }


def calculate_semantic_similarity(predictions: List[str], references: List[str]) -> float:
    """
    Calculate simple lexical overlap similarity.
    This is a basic implementation - for better semantic similarity,
    consider using sentence transformers or other embedding-based methods.
    
    Args:
        predictions: List of generated responses
        references: List of reference responses
        
    Returns:
        Average similarity score
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")
    
    similarities = []
    
    for pred, ref in zip(predictions, references):
        pred_words = set(pred.lower().split())
        ref_words = set(ref.lower().split())
        
        if not pred_words and not ref_words:
            similarities.append(1.0)
        elif not pred_words or not ref_words:
            similarities.append(0.0)
        else:
            intersection = pred_words.intersection(ref_words)
            union = pred_words.union(ref_words)
            similarity = len(intersection) / len(union)
            similarities.append(similarity)
    
    return sum(similarities) / len(similarities)


def calculate_perplexity(predictions: List[str], model, tokenizer) -> float:
    """
    Calculate perplexity of generated responses.
    
    Args:
        predictions: List of generated responses
        model: Language model
        tokenizer: Tokenizer
        
    Returns:
        Average perplexity
    """
    import torch
    
    model.eval()
    total_log_likelihood = 0
    total_tokens = 0
    
    with torch.no_grad():
        for response in predictions:
            if not response.strip():
                continue
                
            # Tokenize
            inputs = tokenizer(response, return_tensors="pt")
            input_ids = inputs["input_ids"]
            
            # Calculate log likelihood
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            
            # Convert loss to log likelihood
            log_likelihood = -loss.item() * input_ids.shape[1]
            total_log_likelihood += log_likelihood
            total_tokens += input_ids.shape[1]
    
    if total_tokens == 0:
        return float('inf')
    
    avg_log_likelihood = total_log_likelihood / total_tokens
    perplexity = math.exp(-avg_log_likelihood)
    
    return perplexity


def calculate_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        predictions: List of generated responses
        references: List of reference responses
        
    Returns:
        Dictionary with all calculated metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics["bleu_4"] = calculate_bleu_score(predictions, references, n_gram=4)
    metrics["bleu_2"] = calculate_bleu_score(predictions, references, n_gram=2)
    metrics["rouge_l"] = calculate_rouge_l(predictions, references)
    metrics["semantic_similarity"] = calculate_semantic_similarity(predictions, references)
    
    # Diversity metrics
    metrics["distinct_1"] = calculate_distinct_ngrams(predictions, n=1)
    metrics["distinct_2"] = calculate_distinct_ngrams(predictions, n=2)
    
    # Length statistics
    length_stats = calculate_response_length_stats(predictions)
    metrics.update(length_stats)
    
    return metrics


def evaluate_responses(
    predictions: List[str],
    references: List[str],
    model=None,
    tokenizer=None
) -> Dict[str, Any]:
    """
    Comprehensive response evaluation.
    
    Args:
        predictions: List of generated responses
        references: List of reference responses
        model: Optional model for perplexity calculation
        tokenizer: Optional tokenizer for perplexity calculation
        
    Returns:
        Dictionary with evaluation results
    """
    results = {
        "num_examples": len(predictions),
        "metrics": calculate_metrics(predictions, references)
    }
    
    # Add perplexity if model is provided
    if model is not None and tokenizer is not None:
        try:
            results["metrics"]["perplexity"] = calculate_perplexity(predictions, model, tokenizer)
        except Exception as e:
            print(f"Warning: Could not calculate perplexity: {e}")
    
    # Add quality assessment
    results["quality_assessment"] = assess_response_quality(predictions, references)
    
    return results


def assess_response_quality(predictions: List[str], references: List[str]) -> Dict[str, Any]:
    """
    Assess overall response quality.
    
    Args:
        predictions: List of generated responses
        references: List of reference responses
        
    Returns:
        Dictionary with quality assessment
    """
    # Count empty or very short responses
    empty_responses = sum(1 for pred in predictions if len(pred.strip()) < 3)
    short_responses = sum(1 for pred in predictions if len(pred.split()) < 3)
    
    # Count repetitive responses
    repetitive_responses = 0
    for pred in predictions:
        words = pred.split()
        if len(set(words)) < len(words) * 0.5:  # Less than 50% unique words
            repetitive_responses += 1
    
    # Calculate relevance (simple keyword overlap)
    relevant_responses = 0
    for pred, ref in zip(predictions, references):
        pred_words = set(pred.lower().split())
        ref_words = set(ref.lower().split())
        overlap = len(pred_words.intersection(ref_words))
        if overlap > 0:
            relevant_responses += 1
    
    total = len(predictions)
    
    return {
        "empty_response_rate": empty_responses / total if total > 0 else 0,
        "short_response_rate": short_responses / total if total > 0 else 0,
        "repetitive_response_rate": repetitive_responses / total if total > 0 else 0,
        "relevant_response_rate": relevant_responses / total if total > 0 else 0,
        "overall_quality_score": (
            (total - empty_responses - repetitive_responses + relevant_responses) / (2 * total)
        ) if total > 0 else 0
    }