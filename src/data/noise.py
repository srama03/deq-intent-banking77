import random # for generating random probabilties
import nltk
from nltk.corpus import stopwords
"""
[1] DROP "NORMAL" WORDS
> text level
> eg dict level
"""
def drop_stopwords(text, noise_level=0.1):
    # add words to new list if not in std stopwords (nltk); else drop randomly w prob of 10%
    stop_words = set(stopwords.words('english'))
    words = text.split()
    result = [word for word in words if not (word in stop_words and random.random()<noise_level)]
    return " ".join(result)

def apply_noise(eg, noise_level=0.1):
    # apply noise as defined in drop_words on a dict level
    eg["text"] = drop_stopwords(eg["text"], noise_level)
    return eg

"""
[2] TYPOS
    2.1. Drop a random letter
    > word level
    > text level
"""
def drop_char(word):
    # take a single word, return it w a random oletter dropped
    if len(word) <= 1:
        return word
    i = random.randint(0, len(word)-1)
    word = word[:i]+word[i+1:]
    return word

def drop_chars_str(text, noise_level=0.1):
    # applies drop_char to random words in a sentence with probability noise_level
    words = text.split()
    result = [drop_char(word) if random.random() < noise_level else word for word in words]
    return " ".join(result)

def apply_typo(eg, noise_level=0.1):
    # apply noise as defined in drop_chars_str on a dict level
    eg["text"] = drop_chars_str(eg["text"], noise_level)
    return eg




def add_noise(data, noise_level=0.1):
    # apply noise on the dataset level by calling apply noise=> use map fn 
    drops = data.map(lambda eg: apply_noise(eg, noise_level)) # lambda to allow change in noise level
    typos = drops.map(lambda eg: apply_typo(eg, noise_level))
    return typos