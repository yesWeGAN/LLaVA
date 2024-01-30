#!/usr/bin/env python
import argparse
import json
import os
from bleu_eval import calculate_and_tabulate_scores
from pprint import pprint

def parse_args():
    """Echo the input arguments to standard output"""
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=100,
        help="Maximum number of samples per class to evaluate.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        help="Directory to store the results.",
    )
    args = parser.parse_args()
    return args


def redo_eval(inputs):
    input_dir = inputs.output_path
    with open(os.path.join(input_dir, "predicted_sentences.json"), "r") as jso:      
        predicted_sentences = json.load(jso)
    with open(os.path.join(input_dir, "reference_sentences.json"), "r") as jso:
        reference_sentences = json.load(jso)
    

    new_preds = {}
    new_refs = {}
    for key, val in predicted_sentences.items():
        new_preds[key] = val[: inputs.limit]
        new_refs[key] = reference_sentences[key][: inputs.limit]
    scores, _ = calculate_and_tabulate_scores(
        reference_sentences=new_refs, predicted_sentences=new_preds
    )
    pprint(scores)

if __name__ == "__main__":
    inputs = parse_args()
    pprint(inputs)
    redo_eval(inputs)


def _restore_predictions_from_combinefile(input_dir):
    """Debug feature in case I'm stupid to overwrite."""
    with open(os.path.join(input_dir, "combined.json"), "r") as jso:
        combined = json.load(jso)

    predicted_sentences = {}
    for key, val in combined.items():
        predicted_sentences[key]=[]
        for k, item in enumerate(val):
            predicted_sentences[key].append(val[k]["pred"])
    with open(os.path.join(input_dir, "predicted_sentences.json"), 'w') as jso:
        json.dump(predicted_sentences, jso)
