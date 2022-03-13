import argparse
from preprocessing.preprocessor import DatasetPreprocessor, DatasetReader
from classifier.train import TrainClassifier
from classifier.predict import PredictClassifier
import datetime
import pandas as pd

pd.options.mode.chained_assignment = None
# TODO: ruwordnet download


def main():
    # EXAMPLE: python main.py preprocess --data ... --model ...
    parser = argparse.ArgumentParser(description='Arguments for semantic shifts')
    parser.add_argument('action', type=str, help='Action type (TRAIN classifier / PREPROCESS data / PREDICT class / PIPELINE)')
    parser.add_argument('--data', type=str, help='Path to dataset', action='append', nargs='+')
    parser.add_argument('--embed_model', type=str, help='Path to model (TF MUSE)')
    parser.add_argument('--class_model', type=str, help='Path to model (Catboost)')
    parser.add_argument('--fname', type=str, help='Name for output file')
    args = parser.parse_args()

    # TODO: не указан доп аргумент

    if args.action.lower() == 'train':
        train_classifier = TrainClassifier(args.data[0][0])
        train_classifier.training(sampling_type='up')
    elif args.action.lower() == 'preprocess':
        if args.data is None:
            raise Exception("Need dataset path to preprocess")
        data_reader = DatasetReader(args.data[0])
        df = data_reader.concat_data()
        preprocessor = DatasetPreprocessor(df, args.embed_model)
        preprocessed_data = preprocessor.fit_transform()
        preprocessed_data.to_csv(f'data/preprocessed_data_{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.csv', index=False)
    elif args.action.lower() == 'predict':
        predict_classifier = PredictClassifier(args.data[0][0], args.class_model)
        predict_classifier.predict_mark()
    elif args.action.lower() == 'pipeline':
        data_reader = DatasetReader(args.data[0])
        df = data_reader.concat_data()
        preprocessor = DatasetPreprocessor(df, args.embed_model)
        preprocessed_data = preprocessor.fit_transform()
        predict_classifier = PredictClassifier(preprocessed_data, args.class_model)
        predict_classifier.predict_mark()
    else:
        pass
        # TODO: error


if __name__ == '__main__':
    main()
