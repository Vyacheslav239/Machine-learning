# coding: utf-8
"""
    Задание на кластеризацию; трюки могут быть сколь угодно грязными --
    а вот код должен быть чистым и прокомментированным предельно ясно.
"""
from collections import defaultdict

import nltk
import os
import numpy as np
import pandas as pd
from typing import List
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering

nltk.download("stopwords")

RUSSIAN_STOPWORDS = set(stopwords.words("russian"))


def process_text(st):
    """ Converting a string of pre-lemmatized words into a list of tokens """
    return [s for s in st.split() if not s.isspace()]


class TextsPairClassifier(object):

    def __init__(self, data: List[str]):
        self.pair_labels = defaultdict(lambda: 0)

        # TODO: you may want to consider other vectorizers? Other approaches to building vectors?
        # Векторизуем наши данные
        vectorizer = TfidfVectorizer(analyzer='word',
                                     ngram_range=(1, 3),
                                     min_df=0.004,
                                     max_df=0.4,
                                     max_features=10000,
                                     stop_words=RUSSIAN_STOPWORDS)
        term_doc_matrix = vectorizer.fit_transform(data)
        # Кластиризуем наши данные, и в качестве расстояни берем Cosine, потому что это расстояние также
        # позволяет наблюдать за углом между векторами.
        # И в качестве меры связи берем average, как самый разумный способ.
        agglomerative_clusterizer = AgglomerativeClustering(linkage='average', n_clusters=6, affinity='cosine')
        self.pair_labels = agglomerative_clusterizer.fit(term_doc_matrix.toarray()).labels_

    def label(self, id1: int, id2: int):
        """ If the items are in the same cluster, return 1, else 0; use self.pair_labels"""
        # todo: ???
        return 1 if self.pair_labels[id1 - 1] == self.pair_labels[id2 - 1] else 0



def generate_submission():
    # reading data
    os.chdir("C:/Users/вячеслав/Desktop/normalized_texts.csv/")
    texts = pd.read_csv("normalized_texts.csv", index_col="id", encoding="utf-8")
    pairs = pd.read_csv("pairs.csv", index_col="id")

    # preparing clusters on object creation and initialization
    classifier = TextsPairClassifier(texts["paragraph_lemmatized"].to_list())

    # generating submission
    with open("submission.csv", "w", encoding="utf-8") as output:
        output.write("id,gold\n")
        for index, id1, id2 in pairs.itertuples():
            result = classifier.label(id1, id2)
            output.write("%s,%s\n" % (index, result))


if __name__ == "__main__":
    generate_submission()
