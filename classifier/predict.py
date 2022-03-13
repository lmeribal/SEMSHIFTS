from .utils import DatasetPreparator
from catboost import CatBoostClassifier
import datetime


class PredictClassifier:
    def __init__(self, data, path_to_model=None):
        self.data = data
        if path_to_model is None:
            path_to_model = 'models/classification_model.cbm'
        self.model = CatBoostClassifier().load_model(path_to_model)
        self.data_columns = ['sim_feature', 'pos_1', 'pos_2', 'sim_pos', 'fuzz', 'cosine']

    def predict_mark(self):
        prepare_dataset = DatasetPreparator(self.data, 'classify')
        dataset = prepare_dataset.preparing()
        preds = self.model.predict(dataset[self.data_columns])
        dataset['pred'] = preds
        return dataset
