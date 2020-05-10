import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

import text_mining.tf_idf.lemmatizer as lm
import text_mining.tokenization.tokenizer as tk

df = pd.read_csv('spam.csv', encoding='latin-1')
df.drop(['Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df.columns = ['target', 'text', 'cleaned_text']
df.drop_duplicates(inplace=True)

for index, row in df.iterrows():
    df['cleaned_text'][index] = [j for i in [lm.lemmatize_sentence(sentence) for sentence in
                                             tk.custom_sentence_tokenize(row['text'])] for j in i]

classes = ['spam', 'ham']

vectorizer = CountVectorizer()
x = vectorizer.fit_transform([' '.join(i) for i in df['cleaned_text']])
y = df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.33,
                                                    random_state=42)

value_counts = y_test.value_counts()
print('Test spam messages: {}'.format(value_counts['spam']))
print('Test ham messages: {}'.format(value_counts['ham']))

classifier = MultinomialNB().fit(x_train, y_train)
y_predicted = classifier.predict(x_test)

print('Confusion Matrix:\n{}'.format(pd.DataFrame(confusion_matrix(y_test, y_predicted, labels=classes),
                                                  index=classes,
                                                  columns=classes)))
print(classification_report(y_test, y_predicted))

print('Accuracy: {}'.format(accuracy_score(y_test, y_predicted)))
