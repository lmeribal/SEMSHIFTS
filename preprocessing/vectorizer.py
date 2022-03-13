import tensorflow_hub as hub
import tensorflow_text


class Vectorizer:
    def __init__(self, dataset, model_path=None):
        self.dataset = dataset
        if model_path is not None:
            self.model = hub.load(model_path)  # TODO больше моделей – Word2Vec / HuggingFace
        else:
            self.model = hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual/3')

    def return_embeds(self):
        first_sense_embeds = self.model([" ".join(el) for el in self.dataset['sense_1']])
        second_sense_embeds = self.model([" ".join(el) for el in self.dataset['sense_2']])
        return [first_sense_embeds, second_sense_embeds]
