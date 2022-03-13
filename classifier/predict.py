from .utils import PrepareDataset
from catboost import CatBoostClassifier
import datetime

# TODO: чтение файла
class PredictClassifier:
    def __init__(self, data, path_to_model=None):
        self.data = data
        if path_to_model is None:
            path_to_model = 'classification_model.cbm'
        self.model = CatBoostClassifier().load_model(path_to_model)
        self.data_columns = ['sim_feature', 'pos_1', 'pos_2', 'sim_pos', 'fuzz', 'cosine']

    def predict_mark(self):
        prepare_dataset = PrepareDataset(self.data, 'classify')
        dataset = prepare_dataset.preparing()
        preds = self.model.predict(dataset[self.data_columns])
        dataset['pred'] = preds
        dataset.to_csv(f'data/preds_{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.csv')

