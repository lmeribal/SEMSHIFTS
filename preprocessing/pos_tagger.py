import nltk
import re
from pymorphy2 import MorphAnalyzer
# from russian_tagsets import converters

class PosTagger:
    def __init__(self):
        self.russian_cached_dict = dict()
        self.russian_morph = MorphAnalyzer()

    def rus_pos_tags_extractor(self, phrase):
        # nltk.download('universal_tagset')
        word = phrase[0].split()[0]
        if word in self.russian_cached_dict:
            parsed = self.russian_cached_dict[word]
        else:
            parsed = self.russian_morph.parse(word)[0].tag.POS
            # to_ud = converters.converter('opencorpora-int', 'ud14')
            # parsed_ud = to_ud(parsed).split(',')[0]
            self.russian_cached_dict[word] = parsed
        return parsed

    @staticmethod
    def eng_pos_tags_extractor(word):
        if re.search('^to ', word[0]):
            return 'VERB'
        else:
            return nltk.pos_tag([word[0]], tagset='universal')[0][1]

    def fit_transform(self, dataset, lang):
        if lang == 'eng':
            dataset['pos_1'] = dataset['sense_1'].apply(self.eng_pos_tags_extractor)
            dataset['pos_2'] = dataset['sense_2'].apply(self.eng_pos_tags_extractor)
        elif lang == 'rus':
            dataset['pos_1'] = dataset['sense_1'].apply(self.rus_pos_tags_extractor)
            dataset['pos_2'] = dataset['sense_2'].apply(self.rus_pos_tags_extractor)
        dataset['sim_pos'] = dataset['pos_1'] == dataset['pos_2']
        return dataset
