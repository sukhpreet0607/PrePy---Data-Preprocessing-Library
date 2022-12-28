import pandas as pd 

def load_data(path): 
    data = pd.read_csv(path, index_col = 0)
    return data


"""
Module that is used to remove or replace all rows with null values in them.
"""
# pylint: disable=R0913
# pylint: disable=W0612
import pandas as pd
import numpy as np


def __find_average(data_frame: pd.DataFrame) -> (dict, dict, dict):
    """
    A helper method to find average values: mean, median
    This method is a helper method for the main No-Null-dataset (NND) method that
    receives a DataFrame in which it find a mean and median for every numeric column.
    :param data_frame: DataFrame to find average of columns of
    :return: returns a tuple consisting of mean and median
    """

    mean = {}
    median = {}
    mode = {}

    for column in list(data_frame):
        if np.issubdtype(data_frame[column].dtype, np.number):
            mean[column] = round(data_frame.loc[:, column].mean(), 2)
            median[column] = round(data_frame.loc[:, column].median(), 2)

        mode[column] = data_frame.loc[:, column].mode()

    return mean, median, mode


def nnd(data_frame, strategy='median', keep_rows=None, remove_rows=None,
        reindex=True, drop=False) -> pd.DataFrame:
    """A method to deal with missing values in the dataset.
    No-null-dataset(NND) is a method that accepts a dataset with null values
    and replaces the rows with missing data with average (median, mean) or removes them.
        :param data_frame: The data to be preprocessed
        :param strategy: The way the method should handle rows with missing values (median,
        mean, mode)
        :param keep_rows: Specify the rows to keep in case 'remove' method was chosen,
        otherwise it has no effect. keep_rows has a priority over remove_rows.
        :param remove_rows: Specify the rows to remove. Can be used with all methods.
        :param reindex: A new dataset will create new indexes if True.
        :param drop: Removes all the rows that contain null values, except for those in keep_rows
        :return: Dataset with no null values
    """

    if keep_rows is None:
        keep_rows = []

    if remove_rows is None:
        remove_rows = []

    if drop:
        remove_rows = data_frame[data_frame.isnull().any(axis=1)].index.values

    remove_rows = [x for x in remove_rows if x not in keep_rows]
    data_after_drop = data_frame.drop(remove_rows, axis=0)

    mean, median, mode = __find_average(data_after_drop)

    for index, row in data_after_drop.iterrows():
        for column, value in row.items():
            if pd.isnull(row[column]):

                if np.issubdtype(data_frame[column].dtype, np.number):

                    if strategy == 'mean':
                        data_after_drop.at[index, column] = mean[column]
                    elif strategy == 'mode':
                        data_after_drop.at[index, column] = mode[column][0]
                    else:
                        data_after_drop.at[index, column] = median[column]

                else:
                    data_after_drop.at[index, column] = mode[column][0]

    no_null_dataset = data_after_drop.reset_index(drop=True) if reindex else data_after_drop

    return no_null_dataset




"""
Module that is used to remove outliers from the dataset with no missing data.
"""
from scipy import stats
import pandas as pd
import numpy as np


def remove_outliers(dataset: pd.DataFrame, strategy='Z', reindex=True, threshold=3) -> \
        pd.DataFrame:
    """
    A method that removes outliers from a dataset that contains no null values. Two strategies
    can be used for outliers' removal: z-score and IQR. In case the dataset contains less than
    12 values only IQR strategy can be used.
    :param dataset: A dataset to remove outliers form containing no null values.
    :param strategy: A strategy for removal (Z or IQR).
    :param reindex: A new dataset will create new indexes if True.
    :param threshold: A threshold for a value to be considered outliers in case Z-score was chosen.
    :return: DataFrame containing no outliers.
    """

    if dataset.count()[0] < 12:
        strategy = 'IQR'

    if strategy.lower() == 'z':

        cols = list(dataset.columns)
        z_scores = pd.DataFrame()

        for col in cols:
            if np.issubdtype(dataset[col].dtype, np.number):
                col_zscore = col + '_zscore'
                z_scores[col_zscore] = np.abs(stats.zscore(dataset[col]))

        # noinspection PyTypeChecker
        no_outliers_dataset = dataset[(z_scores < threshold).all(axis=1)]

    else:

        first_quartile = dataset.quantile(0.25)
        third_quartile = dataset.quantile(0.75)
        iqr = third_quartile - first_quartile

        # noinspection PyTypeChecker
        no_outliers_dataset = dataset[~((dataset < (third_quartile - 1.5 * iqr))
                                        | (dataset > (third_quartile + 1.5 * iqr))).any(axis=1)]

    no_outliers_dataset = no_outliers_dataset.reset_index(
        drop=True) if reindex else no_outliers_dataset

    return no_outliers_dataset





import pandas as pd

def remove_duplicated(data): 
    Dup_Rows = data[data.duplicated()]
    print("\n\nDuplicate Rows : \n {}".format(Dup_Rows))
    DF_RM_DUP = data.drop_duplicates(keep='first')
    print('\n\nResult DataFrame after duplicate removal :\n', DF_RM_DUP.head(n=5))
    
    
    
    
def identify_missing(data, threshold): 
    ''' Identify the features with threshold % missing values 
    Return list of features with to threshold % missing values'''
    # Get the % of missing values for each feature
    missing = data.isnull().sum()/data.shape[0]
    missing_result = pd.DataFrame(missing).reset_index().rename(columns = {'index' : 'features', 0 :'missing_frac'})
    print(missing_result)
    missing_result = missing_result.sort_values(by ='missing_frac', ascending = False)
    
    # Features with threshold missing values
    missing_thres = missing_result[missing_result.missing_frac > threshold]
    features_to_drop = missing_thres.columns
    
    return features_to_drop