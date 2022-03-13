from rapidfuzz import fuzz
from scipy.spatial.distance import cosine
import re
import pandas as pd
from .wordnet import WordnetFeatureExtractor
from .pos_tagger import PosTagger
from .vectorizer import Vectorizer


# TODO: Наследовани Reader и Prepair от этого
class Dataset:
    def __init(self, path_to_dataset):
        self.path_to_dataset = path_to_dataset

    @staticmethod
    def is_valid_data(dataset):
        if 'sense_1' not in dataset.columns or 'sense_2' not in dataset.columns:
            return 0
        elif len(dataset) == 0:
            return 0
        return 1


class DatasetReader:
    def __init__(self, path_to_dataset):
        self.path_to_dataset = path_to_dataset

    @staticmethod
    def is_valid_data(dataset):
        if 'sense_1' not in dataset.columns or 'sense_2' not in dataset.columns:
            return 0
        elif len(dataset) == 0:
            return 0
        return 1

    def concat_data(self):
        # TODO: аргументы для чтения
        result_df = pd.DataFrame()
        for path in self.path_to_dataset:
            data = pd.read_csv(path)
            if self.is_valid_data(data):
                result_df = result_df.append(data)
            else:
                raise Exception("Invalid data format")
        return result_df


class DataPreprocessor:
    def __init__(self, dataset, path_to_model):
        self.dataset = dataset
        self.path_to_model = path_to_model

    @staticmethod
    def word_cleaner(string):
        string = re.sub(' \([\w\W]*\)', '', string)
        string = "".join(char for char in string if char not in ['<', '>'])
        string = re.split(', |/ |/', string)
        return string

    @staticmethod
    def fuzzy_distance(dataset):
        distances_lst = []
        for word_1, word_2 in zip(dataset['sense_1'], dataset['sense_2']):
            distances_lst.append(fuzz.ratio(word_1, word_2))
        dataset['fuzz'] = distances_lst
        return dataset

    @staticmethod
    def cosine_similarity(dataset, embeds):
        distances_lst = []
        for word_1, word_2 in zip(embeds[0], embeds[1]):
            distances_lst.append(1 - cosine(word_1, word_2))
        dataset['cosine'] = distances_lst
        return dataset

    @staticmethod
    def eng_or_rus(text):
        if re.search('[a-zA-Z]', text) and not re.search('(II)+', text):
            return 'eng'
        if re.search('[а-яА-Я]', text):
            return 'rus'
        return 'unknown'

    def fit_transform(self):
        self.dataset['lang'] = self.dataset['sense_1'].apply(self.eng_or_rus)
        self.dataset['sense_1'] = self.dataset['sense_1'].apply(self.word_cleaner)
        self.dataset['sense_2'] = self.dataset['sense_2'].apply(self.word_cleaner)
        dataset_eng = self.dataset[self.dataset['lang'] == 'eng']
        dataset_rus = self.dataset[self.dataset['lang'] == 'rus']
        wn_extractor = WordnetFeatureExtractor()
        pos_tagger = PosTagger()

        if len(dataset_eng) > 0:
            dataset_eng = wn_extractor.fit_transform(dataset_eng, 'eng', ['sense_1', 'sense_2'])
            dataset_eng = pos_tagger.fit_transform(dataset_eng, 'eng')
        if len(dataset_rus) > 0:
            dataset_rus = wn_extractor.fit_transform(dataset_rus, 'rus', ['sense_1', 'sense_2'])
            dataset_rus = pos_tagger.fit_transform(dataset_rus, 'rus')
        preprocessed_dataset = dataset_eng.append(dataset_rus)
        preprocessed_dataset = self.fuzzy_distance(preprocessed_dataset)

        vectorizer = Vectorizer(preprocessed_dataset, self.path_to_model)
        embeds = vectorizer.return_embeds()

        preprocessed_dataset = self.cosine_similarity(preprocessed_dataset, embeds)
        return preprocessed_dataset

