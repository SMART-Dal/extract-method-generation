import re
from collections import Counter
from math import isclose

def tokenize(sentence):
    """Tokenize a sentence into words."""
    return re.findall(r'\b\w+\b', sentence.lower())

def compute_precision_recall(reference_tokens, translation_tokens):
    """Compute precision and recall between reference and translation."""
    common = Counter(reference_tokens) & Counter(translation_tokens)
    precision = sum(common.values()) / len(translation_tokens) if len(translation_tokens) > 0 else 0
    recall = sum(common.values()) / len(reference_tokens) if len(reference_tokens) > 0 else 0
    return precision, recall

def compute_meteor_score(ref_file, trans_file):
    """Compute METEOR score for the entire corpus."""
    total_precision = 0
    total_recall = 0

    with open(ref_file, 'r') as ref_fh, open(trans_file, 'r') as trans_fh:
        reference_corpus = ref_fh.readlines()
        translation_corpus = trans_fh.readlines()

        for reference_sentence, translation_sentence in zip(reference_corpus, translation_corpus):
            reference_tokens = tokenize(reference_sentence)
            translation_tokens = tokenize(translation_sentence)

            precision, recall = compute_precision_recall(reference_tokens, translation_tokens)

            total_precision += precision
            total_recall += recall

    avg_precision = total_precision / len(reference_corpus)
    avg_recall = total_recall / len(reference_corpus)

    if isclose(avg_precision, 0) or isclose(avg_recall, 0):
        f1_score = 0
    else:
        f1_score = 2 * ((avg_precision * avg_recall) / (avg_precision + avg_recall))

    return f1_score

if __name__=="__main__":
    reference = "/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/CodeT5/CodeT5/evaluator/ref.gold"
    translation = "/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/CodeT5/CodeT5/evaluator/trns.pred"
    
    # Preprocess the sentences
    # reference = reference.lower().split()
    # translation = translation.lower().split()

    # Compute the rouge score
    meteor_score = compute_meteor_score(reference, translation)
    
    print(f"Meteor score: {meteor_score}")