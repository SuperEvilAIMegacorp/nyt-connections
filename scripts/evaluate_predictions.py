#!/usr/bin/env python3
"""
Evaluate puzzle solving success for all models and predictions
Outputs results to CSV file
"""

import json
import re
import csv
from pathlib import Path
from collections import defaultdict
import argparse


def do_two_lists_have_same_elements(list1, list2):
    """Check if two lists have the same elements (case-insensitive)"""
    # to lowercase
    list1 = [x.lower() for x in list1]
    list2 = [x.lower() for x in list2]
    if len(list1) != len(list2):
        return False
    return set(list1) == set(list2)


def extract_groups_from_text(text):
    """
    Extract 4 groups of 4 words from formatted text
    Handles formats like:
    - **GROUP NAME**: word1, word2, word3, word4
    - GROUP NAME: word1, word2, word3, word4
    - [word1, word2, word3, word4]
    """
    if not text:
        return []

    # Remove <think> tags if present
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    groups = []

    # Try different patterns
    # Pattern 1: **GROUP**: word1, word2, word3, word4
    pattern1 = r'\*\*[^:*]+\*\*:\s*([^\n]+)'
    matches1 = re.findall(pattern1, text)

    # Pattern 2: GROUP: word1, word2, word3, word4 (without **)
    pattern2 = r'^[A-Z][^:]+:\s*([^\n]+)'
    matches2 = re.findall(pattern2, text, re.MULTILINE)

    # Pattern 3: [word1, word2, word3, word4]
    pattern3 = r'\[(.*?)\]'
    matches3 = re.findall(pattern3, text)

    # Pattern 4: - Group name: word1, word2, word3, word4 (DeepSeek format)
    pattern4 = r'^-\s*[^:]+:\s*([^\n]+)'
    matches4 = re.findall(pattern4, text, re.MULTILINE)

    # Use whichever pattern found 4 groups
    matches = matches1 if len(matches1) >= 4 else (
        matches4 if len(matches4) >= 4 else (
            matches2 if len(matches2) >= 4 else matches3
        )
    )

    # Take only first 4 matches
    matches = matches[:4] if matches else []

    for match in matches:
        # Split by comma and clean up
        words = match.split(',')
        cleaned = []
        for word in words:
            # Remove various artifacts
            word = word.strip().strip('\'"')
            word = re.sub(r'^[0-9]+\.\s*', '', word)  # Remove "1. ", "2. " etc
            word = word.replace('<eos>', '')
            word = word.split('(')[0].strip()  # Remove text in parentheses
            word = word.split('//')[0].strip()  # Remove text after //
            word = word.split(' - ')[0].strip()  # Remove text after dash

            # Handle colon - take the part with more commas
            if ':' in word:
                parts = word.split(':')
                word = max(parts, key=lambda p: p.count(','))

            # Remove text before period if present
            if '.' in word:
                parts = word.split('.', 1)
                if len(parts) > 1:
                    word = parts[1].strip()

            word = word.strip()
            if word:
                cleaned.append(word)

        # Take first 4 words
        if cleaned:
            groups.append(cleaned[:4])

    return groups


def score_puzzle(predicted_groups, ground_truth_groups):
    """
    Score a single puzzle by comparing predicted groups to ground truth
    Returns score (0.0 to 1.0) and number of correct groups (0-4)
    """
    if not predicted_groups or not ground_truth_groups:
        return 0.0, 0

    score = 0.0
    correct_groups = 0

    # For each ground truth group, check if any predicted group matches
    for gt_group in ground_truth_groups:
        if len(gt_group) != 4:
            continue

        for pred_group in predicted_groups:
            if len(pred_group) != 4:
                continue

            if do_two_lists_have_same_elements(pred_group, gt_group):
                score += 0.25
                correct_groups += 1
                break

    return score, correct_groups


def evaluate_predictions_file(filepath, results_dict):
    """
    Evaluate a single predictions JSON file
    Returns dict of puzzle_id -> (score, correct_groups)
    """
    print(f"\nEvaluating: {filepath.name}")

    if not filepath.exists():
        print(f"  File not found, skipping")
        return {}

    with open(filepath, 'r') as f:
        predictions = json.load(f)

    scores = {}
    total_score = 0
    total_count = 0
    perfect_puzzles = 0

    for pred in predictions:
        puzzle_id = pred.get('puzzle_id', 'unknown')
        prediction_text = pred.get('prediction', '')
        ground_truth_text = pred.get('ground_truth', '')

        # Extract groups
        predicted_groups = extract_groups_from_text(prediction_text)
        ground_truth_groups = extract_groups_from_text(ground_truth_text)

        # Score
        score, correct_groups = score_puzzle(predicted_groups, ground_truth_groups)

        scores[puzzle_id] = {
            'score': score,
            'correct_groups': correct_groups,
            'total_groups': 4
        }

        total_score += score
        total_count += 1
        if score == 1.0:
            perfect_puzzles += 1

    # Print summary
    if total_count > 0:
        avg_score = total_score / total_count
        print(f"  Total puzzles: {total_count}")
        print(f"  Average score: {avg_score:.4f} ({avg_score*100:.2f}%)")
        print(f"  Perfect puzzles: {perfect_puzzles}/{total_count} ({perfect_puzzles/total_count*100:.2f}%)")

    return scores


def main():
    parser = argparse.ArgumentParser(description='Evaluate all prediction files')
    parser.add_argument('--predictions-dir', default='data/predictions',
                       help='Directory containing prediction JSON files')
    parser.add_argument('--output', default='data/evaluation_results.csv',
                       help='Output CSV file path')
    args = parser.parse_args()

    predictions_dir = Path(args.predictions_dir)

    print("="*80)
    print("NYT Connections Predictions Evaluation")
    print("="*80)

    # Find all prediction JSON files
    prediction_files = list(predictions_dir.glob("*.json"))

    if not prediction_files:
        print(f"No prediction files found in {predictions_dir}")
        return

    print(f"\nFound {len(prediction_files)} prediction files")

    # Evaluate each file
    all_results = {}

    for filepath in sorted(prediction_files):
        model_name = filepath.stem  # e.g., "exp1_baseline" or "deepseek_deepseek_reasoner_validation"
        scores = evaluate_predictions_file(filepath, all_results)
        all_results[model_name] = scores

    # Write results to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Writing results to: {output_path}")

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'model_name',
            'puzzle_id',
            'score',
            'correct_groups',
            'total_groups'
        ])

        # Write all results
        for model_name, scores in sorted(all_results.items()):
            for puzzle_id, result in sorted(scores.items()):
                writer.writerow([
                    model_name,
                    puzzle_id,
                    result['score'],
                    result['correct_groups'],
                    result['total_groups']
                ])

    # Also create summary CSV
    summary_path = output_path.parent / "evaluation_summary.csv"
    print(f"Writing summary to: {summary_path}")

    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow([
            'model_name',
            'total_puzzles',
            'average_score',
            'percentage',
            'perfect_puzzles',
            'perfect_percentage',
            'total_correct_groups'
        ])

        for model_name, scores in sorted(all_results.items()):
            if not scores:
                continue

            total_score = sum(s['score'] for s in scores.values())
            total_puzzles = len(scores)
            perfect_count = sum(1 for s in scores.values() if s['score'] == 1.0)
            total_correct_groups = sum(s['correct_groups'] for s in scores.values())
            avg_score = total_score / total_puzzles if total_puzzles > 0 else 0

            writer.writerow([
                model_name,
                total_puzzles,
                f"{avg_score:.4f}",
                f"{avg_score*100:.2f}",
                perfect_count,
                f"{perfect_count/total_puzzles*100:.2f}" if total_puzzles > 0 else "0.00",
                total_correct_groups
            ])

    print(f"\n{'='*80}")
    print("Evaluation complete!")
    print(f"Detailed results: {output_path}")
    print(f"Summary results: {summary_path}")
    print("="*80)


if __name__ == '__main__':
    main()
