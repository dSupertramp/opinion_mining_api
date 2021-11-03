import nltk
import html
import re
import json
import pickle
import spacy
import pandas as pd
from textblob import TextBlob

nlp = spacy.load("en_core_web_sm")


def correct_textblob_negation(sentence):
    s = sentence.split()
    if 'not' in s:
        s = TextBlob(sentence)
        if s.sentiment.polarity < 0:
            return s.sentiment.polarity
        else:
            sentence = sentence.split()
            index_not = sentence.index("not")
            next_index = sentence.index("not") + 1
            sentence[index_not], sentence[next_index] = sentence[next_index], sentence[index_not]
            sentence = ' '.join(sentence)
            return TextBlob(sentence).sentiment.polarity
    else:
        return TextBlob(sentence).sentiment.polarity


with open('dict.pk', 'rb') as f:
    DICT = pickle.load(f)


url_pattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"


def process_sentence(sentence):
    sentence = sentence.lower()
    sentence = html.unescape(sentence)
    sentence = sentence.replace(" '", "'")
    sentence = sentence.replace(" n't", "n't")
    sentence = re.sub(url_pattern, '', sentence)
    sentence = re.sub('\[.*?\]', ' ', sentence)
    sentence = re.sub(r'[^\w\d\s\']+', '', sentence)
    sentence = re.sub(r'\w+\d+', '', sentence)
    sentence = re.sub("\S*\d\S*", "", sentence)
    sentence = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", sentence)
    sentence = sentence.replace('\n', '')
    sentence = sentence.replace("''", '')
    sentence = re.sub(' +', ' ', sentence)
    sentence = sentence.strip()
    return sentence


def convert_abbreviations(sentence):
    w = []
    words = sentence.split()
    t = [DICT[w] if w in DICT.keys() else w for w in words]
    return ' '.join(t)


def extract_from_single_opinion(tokenized_sentence):
    Q = []
    for sentence in tokenized_sentence:
        entity_expression = ''
        aspect_expression = ''
        description = ''
        doc = nlp(sentence)
        for index, token in enumerate(doc):
            if (token.pos_ == 'NOUN' and token.dep_ == 'pobj') or (token.dep_ == 'compound'):
                entity_expression = token.text
            try:
                if (token.pos_ == 'NOUN' and token.dep_ == 'nsubj'):
                    aspect_expression = token.text
            except:
                if (token.pos_ == 'NOUN'):
                    aspect_expression = token.text
            if token.pos_ == 'ADJ':
                if str(doc[index-1].pos_) == 'ADV':
                    description = str(doc[index-1]) + ' ' + token.text
                    if str(doc[index-2].dep_) == 'neg':
                        description = str(doc[index-2]) + ' ' + \
                            str(doc[index-1]) + ' ' + token.text
                elif str(doc[index-1].dep_) == 'neg':
                    description = str(doc[index-1]) + ' ' + token.text
                else:
                    description = token.text
                if description == 'other' or description == 'specific':
                    description = ''
            sentiment = correct_textblob_negation(description)
            sentiment_tag = 'positive' if sentiment > 0 else (
                'negative' if sentiment < 0 else 'neutral')
            Q.append({
                'sentence': sentence,
                'entity_expressions': entity_expression,
                'aspect_expressions': aspect_expression,
                'descriptions': description,
                'sentiment_values': sentiment,
                'sentiment_tags': sentiment_tag,
            })
    om = pd.DataFrame.from_dict(Q)
    om = om.drop_duplicates()
    om.drop(om[(om.descriptions == '')].index, inplace=True)
    om.drop(om[(om.aspect_expressions == '')].index, inplace=True)
    om = om.drop_duplicates('descriptions', keep='first')
    om = om[['sentence', 'entity_expressions', 'aspect_expressions',
             'descriptions', 'sentiment_tags', 'sentiment_values']]
    om = om.reset_index(drop=True)
    return om


def process_review(review):
    review_processed = review.replace('.', '. ')
    review_processed = nltk.sent_tokenize(review_processed)
    review_processed = [convert_abbreviations(
        process_sentence(i)) for i in review_processed]
    x = extract_from_single_opinion(review_processed)
    x.sentence = review.replace('\n', '')

    parsed = x.to_json(orient="records")
    json.dumps(parsed, indent=4)
    result = json.loads(parsed)
    return result
