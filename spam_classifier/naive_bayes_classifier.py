import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import text_mining.tf_idf.lemmatizer as lm
import text_mining.tokenization.tokenizer as tk
import text_mining.classification.n_gram_classifier as clf


def get_distance(_pretrained_series, _current_series):
    _distance = 0
    for _word in _current_series.index:
        if _word in _pretrained_series.index:
            _distance += abs(_pretrained_series[_word] - _current_series[_word])
    return _distance


def get_word_frequency(_df, message=False):
    _series = pd.Series()
    if message:
        for _word in [word for sentence in _df for word in sentence]:
            if _word not in _series:
                _series[_word] = 1
            else:
                _series[_word] += 1
    else:
        for _item in _df:
            for _word in [word for sentence in _item for word in sentence]:
                if _word not in _series:
                    _series[_word] = 1
                else:
                    _series[_word] += 1
    return _series.sort_values(ascending=False)


df = pd.read_csv('spam.csv', encoding='latin-1')
df.drop(['Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df.columns = ['target', 'text', 'cleaned_text']

for index, row in df.iterrows():
    df['cleaned_text'][index] = [lm.lemmatize_sentence(sentence) for sentence in tk.custom_sentence_tokenize(row['text'])]

df_train, df_test = train_test_split(df, test_size=0.33, random_state=42)

df_spam_train = df[df['target'] == 'spam']
df_ham_train = df[df['target'] == 'ham']
spam_frequency = get_word_frequency(df_spam_train.loc[:, 'cleaned_text'])
ham_frequency = get_word_frequency(df_ham_train.loc[:, 'cleaned_text'])


classes = ['spam', 'ham']
result_df = pd.DataFrame(np.zeros([len(classes), len(classes)]),
                         index=classes,
                         columns=classes)
spam_count, ham_count = sum([1 for target in df_test.loc[:, 'target'] if target == 'spam']), \
                        sum([1 for target in df_test.loc[:, 'target'] if target == 'ham'])
print('Test spam messages: {}'.format(spam_count))
print('Test ham messages: {}'.format(ham_count))

for index, row in df_test.iterrows():
    text = df_test['cleaned_text'][index]
    target = df_test['target'][index]

    message_frequency = get_word_frequency(text, message=True)
    spam_distance = clf.get_distance(spam_frequency, message_frequency)
    ham_distance = clf.get_distance(ham_frequency, message_frequency)
    predicted = 'spam' if spam_distance > ham_distance else 'ham'

    if predicted is target:
        result_df[predicted][predicted] += 1
    else:
        result_df[predicted][target] += 1

recall = result_df.loc['spam']['spam'] / sum(result_df['spam'])
precision = result_df.loc['spam']['spam'] / sum(result_df.loc['spam'])
f1_score = 2 * precision * recall / (precision + recall)
positive = (result_df.loc['spam']['spam'] + result_df.loc['ham']['ham'])
negative = (result_df.loc['spam']['ham'] + result_df.loc['ham']['spam'])
accuracy = positive / (positive + negative)

print('Resulted dataframe: \n{}'.format(result_df))
print('Recall: {}'.format(recall))
print('Precision: {}'.format(precision))
print('F1-score: {}'.format(f1_score))
print('Accuracy: {}'.format(accuracy))
