import pandas as pd


# TODO: Наследовани Reader и Prepair от этого
class Dataset:
    def __init__(self, data):
        self.data = data

    @staticmethod
    def is_valid_data(dataset, column_lst=None, available_values=None):
        if len(dataset) == 0:
            return 0
        if column_lst:
            for col in column_lst:
                if col not in dataset.columns:
                    return 0
        if available_values:
            for col in available_values.keys():
                if len(set(dataset[col]).difference(set(available_values[col]))) > 0:
                    return 0
        return 1

    def read_data(self, concat=True):
        # TODO: аргументы для чтения
        if concat:
            result_df = pd.DataFrame()
            for path in self.data:
                df = pd.read_csv(path)
                if self.is_valid_data(df):
                    result_df = result_df.append(df)
                else:
                    raise Exception("Invalid data format")
        else:
            if isinstance(self.data, pd.DataFrame):
                result_df = self.data.copy()
            elif isinstance(self.data, str):
                result_df = pd.read_csv(self.data)
            else:
                raise Exception("Invalid data format")
        return result_df
