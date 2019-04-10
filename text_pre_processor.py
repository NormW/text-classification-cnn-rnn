import re
import string
import nltk.data
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def split_text(text):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    array_text = tokenizer.tokenize(text)
    result = list()
    for text in array_text:
        if len(text) > 1:
            result.append(text.lower().strip())
    return result


def stem_text(text):
    stemmer = PorterStemmer()
    # FIXME
    # '... it will sometimes just return a couple minutes late" -> "..it will sometimes just return a couple minutes l"
    return stemmer.stem(" " + text + " ")


def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(text)


def simplify_text(text):
    remove_number = ' \d+'
    process_text = re.sub(remove_number, '', text)
    process_text = process_text.translate(str.maketrans('', '', string.punctuation))
    return process_text


def tokenize_word(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    result = [i for i in tokens if not i in stop_words]
    return result


def text_to_array(text):
    result = list()
    array_text = split_text(text)
    for txt in array_text:
        output = stem_text(txt)
        output = lemmatize_text(output)
        output = simplify_text(output)
        output = tokenize_word(output)
        result.append(output)
    return result




