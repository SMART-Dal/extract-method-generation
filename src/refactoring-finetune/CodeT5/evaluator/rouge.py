import collections
import math

def _get_ngrams(segment, max_order):
    """Extracts all n-grams up to a given maximum order from an input segment."""
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i+order])
            ngram_counts[ngram] += 1
    return ngram_counts

def _lcs_length(reference, translation):
    """Computes the length of the longest common subsequence between a reference and a translation."""
    table = [[0] * (len(translation) + 1) for _ in range(len(reference) + 1)]
    for i, ref_token in enumerate(reference, 1):
        for j, trans_token in enumerate(translation, 1):
            if ref_token == trans_token:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    return table[-1][-1]

def compute_rouge(ref_file, trans_file, max_order=2, smooth=False):
    ref_files = [ref_file]
    reference_text = []
    for reference_filename in ref_files:
        with open(reference_filename) as fh:
            reference_text.append(fh.readlines())
    per_segment_references = []
    for references in zip(*reference_text):
        reference_list = []
        for reference in references:
            reference_list.append(reference.strip().split())
        per_segment_references.append(reference_list)
    translations = []
    with open(trans_file) as fh:
        for line in fh:
            translations.append(line.strip().split())

    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order

    for (references, translation) in zip(per_segment_references, translations):
        for reference in references:
            reference_length = len(reference)
            translation_length = len(translation)

            reference_ngram_counts = _get_ngrams(reference, max_order)
            translation_ngram_counts = _get_ngrams(translation, max_order)

            overlap = sum((reference_ngram_counts & translation_ngram_counts).values())
            for order in range(max_order):
                possible_matches = max(reference_length - order + 1, 0)
                matches_by_order[order] += overlap
                possible_matches_by_order[order] += possible_matches

    precisions = [0] * max_order
    for i in range(max_order):
        if smooth:
            precisions[i] = (matches_by_order[i] + 1.) / (possible_matches_by_order[i] + 1.)
        else:
            precisions[i] = matches_by_order[i] / possible_matches_by_order[i] if possible_matches_by_order[i] > 0 else 0

    rouge_n = sum(precisions) / max_order

    # Compute ROUGE-L
    matches = 0
    possible_matches = 0

    for (references, translation) in zip(per_segment_references, translations):
        for reference in references:
            matches += _lcs_length(reference, translation)
            possible_matches += len(reference)

    if smooth:
        rouge_l = (matches + 1.) / (possible_matches + 1.)
    else:
        rouge_l = matches / possible_matches if possible_matches > 0 else 0

    return rouge_n, rouge_l

if __name__=="__main__":
    reference = "/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/CodeT5/CodeT5/evaluator/ref.gold"
    translation = "/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/CodeT5/CodeT5/evaluator/trns.pred"
    
    # Preprocess the sentences
    # reference = reference.lower().split()
    # translation = translation.lower().split()

    # Compute the rouge score
    rouge_score = compute_rouge(reference, translation, max_order=4, smooth=False)
    
    print(f"Rouge score: {rouge_score}")
