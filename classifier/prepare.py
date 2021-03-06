from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from utils.utlis import Dataset


class DatasetPreparator(Dataset):
    # TODO: разделить класс на utils от которого будут наследоваться класс для подготовки датасета тренировки и предикта
    def __init__(self, data, type, train_size=0.67, sampling_type=None):
        # type – for train or for classify?
        self.train_size = train_size
        self.type = type
        self.sampling_type = sampling_type # up or down or None
        self.classify_data_columns = ['sim_feature', 'pos_1', 'pos_2', 'sim_pos', 'fuzz', 'cosine']
        self.train_data_columns = self.classify_data_columns + ['mark']
        self.available_values = {
            'mark': [0, 1, 2]
        }
        super(DatasetPreparator, self).__init__(data)

    @staticmethod
    def get_majority_class(y_train):
        return y_train.value_counts().index[0]

    def sampling(self, X_train, y_train, sampling_type):
        majority_class = self.get_majority_class(y_train)
        if majority_class == 1:
            minority_class = 0
        else:
            minority_class = 1
        total_df = X_train.copy()
        total_df['mark'] = y_train
        majority_df = total_df[total_df['mark'] == majority_class]
        minority_df = total_df[total_df['mark'] == minority_class]
        majority_length = len(majority_df)
        minority_length = len(minority_df)
        if sampling_type == 'down':
            majority_downsampled_df = resample(majority_df,
                                               replace=True,
                                               n_samples=minority_length,
                                               random_state=42)
            X_train = majority_downsampled_df.append(minority_df)
        elif sampling_type == 'up':
            minority_upsampled_df = resample(minority_df,
                                             replace=True,
                                             n_samples=majority_length,
                                             random_state=42)
            X_train = minority_upsampled_df.append(majority_df)
        y_train = X_train['mark'].copy()
        X_train = X_train[self.classify_data_columns]
        return X_train, y_train

    def preparing(self):
        dataset = super().read_data(concat=False)
        if self.type == 'classify':
            if not self.is_valid_data(dataset,
                                      column_lst=self.classify_data_columns):
                raise Exception("Invalid data format")
        elif self.type == 'train':
            if not self.is_valid_data(dataset,
                                      column_lst=self.train_data_columns,
                                      available_values=self.available_values):
                raise Exception("Invalid data format")
        for col in ['pos_1', 'pos_2']:
            dataset[col] = dataset[col].astype(str).astype('category').cat.codes
        if self.type == 'train':
            dataset = dataset[dataset['mark'] != 2]
            X = dataset[self.classify_data_columns]
            Y = dataset['mark']
            X_train, X_test, y_train, y_test = train_test_split(X,
                                                                Y,
                                                                train_size=self.train_size,
                                                                shuffle=True,
                                                                random_state=42)
            if self.sampling_type is not None:
                X_train, y_train = self.sampling(X_train, y_train, self.sampling_type)
            return X_train, X_test, y_train, y_test
        elif self.type == 'classify':
            return dataset
