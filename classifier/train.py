import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import scikitplot as skplt
import matplotlib.pyplot as plt
from .utils import PrepareDataset


class TrainClassifier:
    def __init__(self, path_to_data, iterations=350, depth=3, eval_metric='F1', l2_leaf_reg=1):
        self.path_to_data = path_to_data
        self.model = CatBoostClassifier(iterations=iterations,
                                        depth=depth,
                                        random_seed=42,
                                        eval_metric=eval_metric,
                                        l2_leaf_reg=l2_leaf_reg)

    def evaluate(self, preds_class, true_class, preds_proba, train_data, feature_importance=True, draw_roc_auc=True):
        print("Classification report:")
        print(classification_report(preds_class, true_class))
        print("F-мера: ", f1_score(preds_class, true_class, average='macro'))
        print("Accuracy: ", accuracy_score(true_class, preds_class))

        if feature_importance:
            print("Feature importance:")
            importance_df = pd.DataFrame({'feature_importance': self.model.get_feature_importance(),
                                          'feature_names': train_data.columns}).sort_values(by=['feature_importance'],
                                                                                            ascending=False)
            print(importance_df.to_markdown())
        if draw_roc_auc:
            skplt.metrics.plot_roc_curve(true_class, preds_proba)
            plt.savefig('roc_auc.png')

    def training(self, train_size=0.67, sampling_type=None):
        # TODO:    нужно как то валидировать данные
        dataset_preparing = PrepareDataset(path_to_data=self.path_to_data,
                                           type='train',
                                           train_size=train_size,
                                           sampling_type=sampling_type)
        X_train, X_test, y_train, y_test = dataset_preparing.preparing()
        self.model.fit(X_train, y_train)
        preds_class = self.model.predict(X_test)
        preds_probas = self.model.predict_proba(X_test)

        self.evaluate(preds_class, y_test, preds_probas, X_train)

        self.model.save_model('classification_model.cbm')
