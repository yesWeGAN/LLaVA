#!/usr/bin/env python
import argparse
import json
import os
import random
import traceback
from collections import Counter
from datetime import datetime
from pprint import pprint
from tqdm import tqdm

import torch

from eval_methods import (
    calculate_corpus_bleu1_filtered,
    calculate_corpus_bleu1_native,
    calculate_mean_bleu1_filtered,
    calculate_mean_bleu1_native,
    check_filler,
    check_high_freq,
    parse_response,
    split_string_filtered,
)
from predict import DatasetInference, Predictor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    """Echo the input arguments to standard output"""
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "-m",
        "--max_tokens",
        type=int,
        default=384,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=100,
        help="Maximum number of samples per class to evaluate.",
    )

    parser.add_argument(
        "-p",
        "--top_p",
        type=float,
        default=0.7,
        help="Sampling top-p.",
    )
    parser.add_argument(
        "-t",
        "--temp",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "-f",
        "--filepath_validation",
        type=str,
        help="JSON-file with validation samples.",
    )
    parser.add_argument(
        "-a", "--alt_prompt", action="store_true", help="Use alternative prompts."
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        help="Directory to store the results.",
    )
    args = parser.parse_args()
    return args


def run_inference_for_eval(inputs):
    """This method sets up the model and runs the inference on the given validation set.
    Args:
        inputs: Namespace object from argparse.
    Returns:
        Three dictionaries containing all prompts, references and predictions."""

    # model setup
    predictor = Predictor()
    predictor.setup(
        model_path="/home/frank/llava/LLaVA/checkpoints/llava-v1.5-7b-task-lora",
        model_base="liuhaotian/llava-v1.5-7b",
        model_name="llava-v1.5-7b-task-lora",
    )
    predictor.model.to(device)
    predictor.model.eval()

    # dataset setup
    val_dataset = DatasetInference(
        json_filepath="/home/frank/ssd/datasets/cropshop/cropshop_v2_3k_val.json",
    )

    # sample indices (inspection of results is more diverse if mixed)
    indices = random.sample(range(len(val_dataset.ds)), len(val_dataset.ds))
    alternative_prompt = True if inputs.alt_prompt else False
    reference_prompts = {}  # store the prompts and image paths
    reference_sentences = {}  # store all reference sentences
    predicted_sentences = {}  # store all predicted sentences
    predicted_classes = {}  # count the number of samples per class
    for index in tqdm(indices):
        try:
            img, item_type, convo = val_dataset.load_sample(index=index)
            if item_type not in predicted_classes.keys():
                predicted_classes[item_type] = 1
                reference_sentences[item_type] = []
                predicted_sentences[item_type] = []
                reference_prompts[item_type] = []
            else:
                predicted_classes[item_type] += 1

            if predicted_classes[item_type] > inputs.limit:
                continue  # do not run prediction if limit for this class is already reached
            else:
                alternative_question = f"Write a detailed fashion store description for the {item_type} in the image."
                # alternative prompt means not passing the item title/name into the prompt. might reduce overfitting.
                if alternative_prompt:
                    response = predictor.predict(
                        image=img,
                        prompt=alternative_question,
                        max_tokens=inputs.max_tokens,
                        temperature=inputs.temp,
                        top_p=inputs.top_p,
                    )
                else:
                    response = predictor.predict(
                        image=img,
                        prompt=convo[0],
                        max_tokens=inputs.max_tokens,
                        temperature=inputs.temp,
                        top_p=inputs.top_p,
                    )

                generation = parse_response(response_generator=response)
                reference_sentences[item_type].append(convo[1])
                predicted_sentences[item_type].append(generation)
                if alternative_prompt:
                    reference_prompts[item_type].append(
                        (alternative_question, img.as_posix())
                    )
                else:
                    reference_prompts[item_type].append((convo[0], img.as_posix()))
        except Exception as e:
            traceback.print_exception(e)
    return reference_prompts, reference_sentences, predicted_sentences


def calculate_and_tabulate_scores(reference_sentences, predicted_sentences):
    """Calculates BLEU-1-scores (both a native version and the weighted fashionBLEU-score).
    Args:
        reference_sentences: Dict of key: item_type, val: actual target descriptions.
        predicted_sentences: Dict of key: item_type, val: predicted target descriptions.
    Returns:
        scores-dict, list: scores printed as a markup-table."""
    bleu_score_native_mean = calculate_mean_bleu1_native(
        refs=reference_sentences, preds=predicted_sentences
    )
    bleu_score_filtered_mean = calculate_mean_bleu1_filtered(
        refs=reference_sentences, preds=predicted_sentences
    )

    bleu_score_native_corp = calculate_corpus_bleu1_native(
        refs=reference_sentences, preds=predicted_sentences
    )
    bleu_score_filtered_corp = calculate_corpus_bleu1_filtered(
        refs=reference_sentences, preds=predicted_sentences
    )
    tabulate_scores = []
    for item_type in [
        "dress",
        "top",
        "pants",
        "skirt",
        "accessoire",
        "one-piece",
        "lingerie",
        "hat",
        "swimwear",
        "footwear",
        "item",
        "sunglasses",
        "underwear",
    ]:
        try:
            print(
                f"{item_type} | {bleu_score_native_corp[item_type]} | {bleu_score_filtered_corp[item_type]} | {bleu_score_native_mean[item_type]} | {bleu_score_filtered_mean[item_type]} | {len(reference_sentences[item_type])}"
            )
            tabulate_scores.append(
                f"{item_type} | {bleu_score_native_corp[item_type]} | {bleu_score_filtered_corp[item_type]} | {bleu_score_native_mean[item_type]} | {bleu_score_filtered_mean[item_type]} | {len(reference_sentences[item_type])}"
            )
        except:
            pass
    result_dict = {}
    result_dict["native_corp"] = bleu_score_native_corp
    result_dict["filtered_corp"] = bleu_score_filtered_corp
    result_dict["native_mean"] = bleu_score_native_mean
    result_dict["filtered_mean"] = bleu_score_filtered_mean
    return result_dict, tabulate_scores


def store_results(
    inputs,
    predicted_sentences,
    reference_sentences,
    reference_prompts,
    bleu_results,
    tabulate_scores,
):
    """Does what it says on the can. Stores everything into the folder given on cmd-line."""
    prompt_style = "alt_prompt" if inputs.alt_prompt else "full_prompt"
    filename_base = f"{prompt_style}_{inputs.max_tokens}_maxtok_{inputs.temp}_temp_{inputs.top_p}_top_p_{inputs.limit}_limit"
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H-%M")
    output_dir = os.path.join(inputs.output_path, f"eval_{filename_base}_{date_time}")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "predicted_sentences.json"), "w") as jso:
        json.dump(predicted_sentences, jso)
    with open(os.path.join(output_dir, "reference_sentences.json"), "w") as jso:
        json.dump(reference_sentences, jso)
    with open(os.path.join(output_dir, "reference_prompts.json"), "w") as jso:
        json.dump(reference_prompts, jso)
    with open(os.path.join(output_dir, "bleu_scores.json"), "w") as jso:
        json.dump(bleu_results, jso)

    with open(os.path.join(output_dir, "tab_scores.txt"), "w") as txout:
        for line in tabulate_scores:
            txout.write(f"{line}\n")

    all_combined = {}
    for key in reference_prompts.keys():
        all_combined[key] = []
        for k, _ in enumerate(reference_prompts[key]):
            all_combined[key].append(
                {
                    "img": reference_prompts[key][k][1],
                    "prompt": reference_prompts[key][k][0],
                    "pred": predicted_sentences[key][k],
                    "ref": reference_sentences[key][k],
                }
            )
    with open(os.path.join(output_dir, "combined.json"), "w") as jso:
        json.dump(all_combined, jso)
    print(f"Done storing results under {output_dir}")


def print_word_frequencies(predicted_sentences, reference_sentences):
    """Debugging feature to print word frequencies in preds/refs to filter words that are frequent, but do not carry information."""
    predicted_words_list = []
    for key, predlist in predicted_sentences.items():
        for pred in predlist:
            predicted_words_list.extend(split_string_filtered(pred))
    counts = Counter(predicted_words_list)
    pprint(list(filter(check_high_freq, counts.items())))

    reference_words_list = []
    for key, predlist in reference_sentences.items():
        for pred in predlist:
            reference_words_list.extend(split_string_filtered(pred))
    counts = Counter(reference_words_list)
    pprint(list(filter(check_high_freq, counts.items())))
    list(filter(check_filler, predicted_words_list))


def main():
    inputs = parse_args()
    pprint(inputs)
    ref_prompts, ref_sentences, pred_sentences = run_inference_for_eval(inputs)
    bleu_scores, tab_scores = calculate_and_tabulate_scores(
        reference_sentences=ref_sentences, predicted_sentences=pred_sentences
    )
    store_results(
        inputs=inputs,
        predicted_sentences=pred_sentences,
        reference_sentences=ref_sentences,
        reference_prompts=ref_prompts,
        bleu_results=bleu_scores,
        tabulate_scores=tab_scores,
    )


if __name__ == "__main__":
    main()
