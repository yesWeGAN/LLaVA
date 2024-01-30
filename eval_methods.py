#!/usr/bin/env python
import nltk
import numpy as np


# some filter functions
def check_filler(word: str) -> bool:
    """Check if a word is only a filler."""
    if word in FILLERS:
        return False
    return True


def check_high_freq(pair: tuple) -> bool:
    """check if a word occurs more than once."""
    word, count = pair
    if count > 2:
        return True
    return False


def split_string(string: str) -> list:
    return string.replace(".", " ").replace("!", " ").replace("-", " ").split()


def split_string_filtered(string: str) -> list:
    """Split the string and filter filler-words."""
    return list(
        filter(
            check_filler,
            [
                elem.lower()
                for elem in string.replace(".", " ")
                .replace("!", " ")
                .replace("-", " ")
                .split()
            ],
        )
    )


def print_response(response_generator):
    generation = "".join(list(response_generator))
    return generation


def parse_response(response_generator):
    """Evaluates the generator object by casting it to list."""
    generation = "".join(list(response_generator))
    return generation


def calculate_mean_bleu1_filtered(refs: dict, preds: dict) -> dict:
    """This function calculates a BLEU-1 score per sample and averages them per class (item_type).
    It uses the filtered reference / truth description, i.e.
    words like this, is, a, and, or, here, .. are ignored as they do not carry information about the item.

    Args:
        refs:   reference sentences per item.
        preds:  prediction sentences per item.
    Returns:
        dict:   containing BLEU-1-scores averaged per class / item-type.
    """
    bleu_scores_per_item = {}
    for item_type, item_descriptions in refs.items():
        item_generations = preds[item_type]
        bleu_scores = []
        for k, ref in enumerate(item_descriptions):
            predicted_sentence = item_generations[k]
            BLEUscore = nltk.translate.bleu_score.sentence_bleu(
                [split_string_filtered(ref)],
                split_string_filtered(predicted_sentence),
                weights=(1, 0, 0, 0),
            )
            bleu_scores.append(BLEUscore)
        avg_bleu_1 = np.mean(np.array(bleu_scores))
        bleu_scores_per_item[item_type] = round(avg_bleu_1, 3)
    return bleu_scores_per_item


def calculate_mean_bleu1_native(refs: dict, preds: dict) -> dict:
    """This function calculates a BLEU-1 score per sample and averages them per class (item_type).
    It uses the full reference / truth description, i.e.
    words like this, is, a, and, or, here, .. are included.

    Args:
        refs:   reference sentences per item.
        preds:  prediction sentences per item.
    Returns:
        dict:   containing BLEU-1-scores averaged per class / item-type.
    """
    bleu_scores_per_item = {}
    for item_type, item_descriptions in refs.items():
        item_generations = preds[item_type]
        bleu_scores = []
        for k, ref in enumerate(item_descriptions):
            predicted_sentence = item_generations[k]
            BLEUscore = nltk.translate.bleu_score.sentence_bleu(
                [split_string(ref)],
                split_string(predicted_sentence),
                weights=(1, 0, 0, 0),
            )
            bleu_scores.append(BLEUscore)
        avg_bleu_1 = np.mean(np.array(bleu_scores))
        bleu_scores_per_item[item_type] = round(avg_bleu_1, 3)
    return bleu_scores_per_item


def calculate_corpus_bleu1_filtered(refs: dict, preds: dict) -> dict:
    """This function calculates a BLEU-1 score on the whole generated corpus per class (item_type).
    It uses the filtered reference / truth description, i.e.
    words like this, is, a, and, or, here, .. are ignored as they do not carry information about the item.

    Args:
        refs:   reference sentences per item.
        preds:  prediction sentences per item.
    Returns:
        dict:   containing BLEU-1-scores on corpus per class / item-type.
    """

    bleu_scores_per_item = {}
    for item_type, item_descriptions in refs.items():
        item_generations = preds[item_type]
        all_refs = []
        all_preds = []
        for k, ref in enumerate(item_descriptions):
            predicted_sentence = item_generations[k]
            all_refs.extend(split_string_filtered(ref))
            all_preds.extend(split_string_filtered(predicted_sentence))
        BLEUscore = nltk.translate.bleu_score.sentence_bleu(
            [all_refs],
            all_preds,
            weights=(1, 0, 0, 0),
        )
        bleu_scores_per_item[item_type] = round(BLEUscore, 3)
    return bleu_scores_per_item


def calculate_corpus_bleu1_native(refs: dict, preds: dict) -> dict:
    """This function calculates a BLEU-1 score on the whole generated corpus per class (item_type).
    It uses the full reference / truth description, i.e.
    words like this, is, a, and, or, here, .. are included.

    Args:
        refs:   reference sentences per item.
        preds:  prediction sentences per item.
    Returns:
        dict:   containing BLEU-1-scores on corpus per class / item-type.
    """
    bleu_scores_per_item = {}
    for item_type, item_descriptions in refs.items():
        item_generations = preds[item_type]
        all_refs = []
        all_preds = []
        for k, ref in enumerate(item_descriptions):
            predicted_sentence = item_generations[k]
            all_refs.extend(split_string(ref))
            all_preds.extend(split_string(predicted_sentence))
        BLEUscore = nltk.translate.bleu_score.sentence_bleu(
            [all_refs],
            all_preds,
            weights=(1, 0, 0, 0),
        )
        bleu_scores_per_item[item_type] = round(BLEUscore, 3)
    return bleu_scores_per_item


FILLERS = [
    "a",
    "the",
    "is",
    "arial",
    "bold",
    "text",
    "that",
    "font",
    "12px",
    "transform",
    "postage",
    "and",
    "with",
    "this",
    "for",
    "of",
    "to",
    "{",
    "or",
    "any",
    "in",
    "it's",
    "that's",
    "also",
    "like",
    "not",
    "too",
    "label+packaging",
    "melbourne",
    "australia",
    "delivery",
    "same",
    "you're",
    "whether",
    "are",
    "has",
    "an",
    "retail",
    "locations",
    "returned",
    "cannot",
    "be",
    "it",
    "all",
    "non",
    "up",
    "more",
    "at",
    "service",
    "gift",
    "wrapping",
    "overnight",
    "24hr",
    "size",
    "care",
    "0;}",
    "p",
    "#888;",
    "capitalize;",
    "bold;",
    "transform:",
    "h3",
    "12px;",
    "size:",
    "serif;",
    "0;",
    "san",
    "helvetica,",
    "arial,",
    "family:",
    "she",
    "&amp",
    "your",
    "will",
    "+",
    "d_content",
    "}",
    "bust,",
    "available",
    "our",
    "you",
    "in",
    "any",
    "out",
    "make",
    "5px",
    "adds",
    "while",
    "made",
    "product",
    "&amp;",
    "color:",
    "#888;",
    ">",
    "5em",
    "label+packaging:",
    "tagged",
    "packaged",
    "labeled",
    "look",
    "d_wrapper",
    "offers",
    "by",
    "vary",
    "next",
    "new",
    "its",
    "need",
    "we're",
    "some",
    "around",
    "must",
    "when",
    "want",
    "take",
    "best",
    "do",
    "try",
    "cotton/polyester",
    "making",
    "go",
    "did",
    "know",
    "so",
    "cotton/acrylic",
    "lightweight",
    "lightweight,",
    "first",
    "move",
    "won't",
    "9",
    "vlogger",
    "very",
    "if",
    "af",
    "me",
    "you",
    "anything",
    "only",
    "well",
    "got",
    "self",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "than",
    "these",
    "'fit",
    "no",
    "ours",
    "from",
    "on",
    "18k",
    "24mm",
    "5mm",
]
