from nltk.corpus import wordnet as eng_wn
from ruwordnet import RuWordNet
import re
import nltk


class WordnetFeatureExtractor:
    @staticmethod
    def is_features_similar(dataset):
        if len(set(dataset['features_1']).intersection(set(dataset['features_2']))) > 0:
            return 1
        return 0

    @staticmethod
    def eng_wordnet_feature_extractor(string):
        string = re.sub('^to ', '', string)
        string = "_".join(string.split())
        lst = []
        s = eng_wn.synsets(string)
        if len(s) == 0:
            return lst
        else:
            for s_synset in s:
                holonyms = s_synset.member_holonyms()
                for h in holonyms:
                    lst.append(h.name().split('.')[0])
                hypernyms = s_synset.hypernyms()
                for h in hypernyms:
                    lst.append(h.name().split('.')[0])
        return list(set(lst))

    @staticmethod
    def rus_wordnet_feature_extractor(string):
        wn = RuWordNet()
        lst = []
        s = wn.get_synsets(string)
        if len(s) == 0:
            return lst
        else:
            for s_synset in s:
                for h in s_synset.hypernyms:
                    title = re.sub(r"(\w+)СЯ", r"\1", h.title)
                    lst.append(title)
                for h in s_synset.holonyms:
                    title = re.sub(r"(\w+)СЯ", r"\1", h.title)
                    lst.append(title)
                for h in s_synset.domains:
                    title = re.sub(r"(\w+)СЯ", r"\1", h.title)
                    lst.append(title)
        return list(set(lst))

    def fit_transform(self, dataset, lang, cols):
        nltk.download('omw-1.4')
        features_dict = {
            'features_1': [],
            'features_2': []
        }
        for sense_arr_1, sense_arr_2 in zip(dataset[cols[0]], dataset[cols[1]]):
            tmp_arr_1 = []
            tmp_arr_2 = []
            for word_1, word_2 in zip(sense_arr_1, sense_arr_2):
                if lang == 'eng':
                    tmp_arr_1.append(self.eng_wordnet_feature_extractor(word_1))
                    tmp_arr_2.append(self.eng_wordnet_feature_extractor(word_2))
                elif lang == 'rus':
                    tmp_arr_1.append(self.rus_wordnet_feature_extractor(word_1))
                    tmp_arr_2.append(self.rus_wordnet_feature_extractor(word_2))
                else:
                    pass  # TODO: raise Exception
            features_dict['features_1'].append([item for sublist in tmp_arr_1 for item in sublist])
            features_dict['features_2'].append([item for sublist in tmp_arr_2 for item in sublist])
        dataset['features_1'] = features_dict['features_1']
        dataset['features_2'] = features_dict['features_2']
        dataset['sim_feature'] = dataset.apply(self.is_features_similar, axis=1)
        return dataset
