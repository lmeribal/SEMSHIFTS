import argparse
from preprocessing.preprocessor import DataPreprocessor, DatasetReader
from classifier.train import TrainClassifier
from classifier.predict import PredictClassifier
import datetime
import pandas as pd
import os

pd.options.mode.chained_assignment = None
# TODO: ruwordnet download
# TODO: requirements.txt
# TODO: пользовательские наименования колонок со смыслами


def main():
    parser = argparse.ArgumentParser(description='Arguments for SEMSHIFTS project')
    parser.add_argument('action', type=str, help='Action type (TRAIN classifier / PREPROCESS data / PREDICT class)')
    parser.add_argument('--data', type=str, help='Path to dataset', action='append', nargs='+')
    parser.add_argument('--embed_model', type=str, help='Path to embedding model (TF MUSE)')
    parser.add_argument('--class_model', type=str, help='Path to classification model (Catboost)')
    parser.add_argument('--sampling_type', type=str, help='Sampling type (up or down)')
    parser.add_argument('--file_name', type=str, help='Name for output file')
    parser.add_argument('--model_name', type=str, help='Name for output classification model')
    parser.add_argument('--col_1_name', type=str, help='Name of first sense column')
    parser.add_argument('--col_2_name', type=str, help='Name of second sense column')
    args = parser.parse_args()

    if args.action.lower() not in ['train', 'preprocess', 'predict', 'cluster', 'train_pipeline', 'predict_pipeline']:
        raise Exception(f"There is no such action: {args.action}")
    else:
        if args.data is None:
            raise Exception("You need to pass the path to the data")
        if not os.path.exists(args.data):
            raise FileNotFoundError(f"No such file or directory: {args.data}")
        if args.sampling_type not in [None, 'up', 'down']:
            raise Exception(f"The sampling_type argument can take the values 'up' or 'down', not {args.sampling_type}")
        if args.embed_model is not None and not os.path.exists(args.embed_model):
            raise FileNotFoundError(f"No such file or directory: {args.embed_model}")
        if args.class_model is not None and not os.path.exists(args.class_model):
            raise FileNotFoundError(f"No such file or directory: {args.class_model}")

    if args.action.lower() == 'train':
        train_classifier = TrainClassifier(args.data[0][0], model_name=args.model_name)
        train_classifier.training(sampling_type=args.sampling_type)
    elif args.action.lower() == 'preprocess':
        data_reader = DatasetReader(args.data[0])
        df = data_reader.concat_data()
        preprocessor = DataPreprocessor(df, args.embed_model)
        preprocessed_data = preprocessor.fit_transform()
        if args.file_name is None:
            fname = f'data/preprocessed_data_{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.csv'
        else:
            fname = args.file_name
        preprocessed_data.to_csv(fname, index=False)
    elif args.action.lower() == 'predict':
        predict_classifier = PredictClassifier(args.data[0][0], args.class_model)
        predict_classifier.predict_mark()
    elif args.action.lower() == 'cluster':
        pass
    elif args.action.lower() == 'train_pipeline':
        pass
    elif args.action.lower() == 'predict_pipeline':
        print("Reading the data...")
        data_reader = DatasetReader(args.data[0])
        df = data_reader.concat_data()
        print("Preprocessing the data...")
        preprocessor = DataPreprocessor(df, args.embed_model)
        preprocessed_data = preprocessor.fit_transform()
        print("Classes predicting...")
        predict_classifier = PredictClassifier(preprocessed_data, args.class_model)
        predicted_data = predict_classifier.predict_mark()
        if args.file_name is None:
            fname = f'data/preds_{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.csv'
        else:
            fname = args.file_name
        predicted_data.to_csv(fname, index=False)


if __name__ == '__main__':
    main()
