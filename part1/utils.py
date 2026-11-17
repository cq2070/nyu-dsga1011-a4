import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    import random
    import re

    text = example["text"]
    rng = random.Random(text)
    sentiment_words = {
        "amazing", "awesome", "beautiful", "brilliant", "enjoyable",
        "excellent", "fantastic", "favorite", "great", "hilarious",
        "loved", "perfect", "superb", "wonderful",
        "annoying", "awful", "boring", "crappy", "disappointing",
        "dreadful", "frustrating", "hated", "horrible", "pathetic",
        "stupid", "terrible", "unwatchable", "worst",
    }

    def transpose(match):
        word = match.group()
        if word.startswith("<"):
            return word

        rate = 0.90 if word.lower() in sentiment_words else 0.35
        if rng.random() >= rate:
            return word

        positions = [
            i for i in range(len(word) - 1)
            if word[i].lower() != word[i + 1].lower()
        ]
        if not positions:
            return word

        i = rng.choice(positions)
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]

    example["text"] = re.sub(r"<[^>]*>|\b[A-Za-z]{5,}\b", transpose, text)

    ##### YOUR CODE ENDS HERE ######

    return example
