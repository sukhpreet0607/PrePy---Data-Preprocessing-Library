import numpy as np 
import pandas as pd 
from sklearn.ensemble import ExtraTreesClassifier

def identify_collinear(data, corr_threshold):
    ''' Identify the most correlated features
    Return dataframe with correlated features '''

    # Correlation matrix
    corr = data.corr()

    # Extraction of the upper triangle of the correlation matrix 
    upper = corr.where(np.triu(np.ones(corr.shape), k = 1).astype(bool))

    # Features with correlation > corr_threshold
    features_to_drop = [column for column in upper.columns if any(upper[column].abs() > corr_threshold)]
    print(features_to_drop)

    # Datafame with pairs of correlated features 
    collinear = pd.DataFrame(columns = ['drop_feature', 'corr_feature', 'corr_value'])
 
    for col in features_to_drop:
        # In upper, we get for each features_to_drop, the correlated features and we save the pair in collinear dataframe 

        # correlated features to features_to_drop  
        corr_features = list(upper.index[upper[col].abs() > corr_threshold])

        # Values 
        corr_values = list(upper[col][upper[col].abs() > corr_threshold])
        
        # Get drop_features
        drop_features = [col for _ in range(len(corr_features))]
        
        # extraction 
        temp_df = pd.DataFrame.from_dict({'drop_feature' : drop_features,'corr_feature' : corr_features, 'corr_value': corr_values})
        
        collinear = collinear.append(temp_df, ignore_index = True)
    
    return collinear



def identify_features_importance(data, label_column, threshold): 
    ''' Identify most important features during a classification 
    return list of the most important features '''

    features = data.drop([label_column], axis = 1)
    labels = data[label_column]
    
    model = ExtraTreesClassifier()
    model.fit(features, labels)
    
    # Get features importances in %
    feature_importances = pd.DataFrame( columns = features.columns)
    feature_importances_values = model.feature_importances_ *100
    feature_importances.loc[len(feature_importances)] = feature_importances_values
    
    # Extaction of the most important features according to threshold : 
    for col in feature_importances:
        if feature_importances.loc[0, col] < threshold:
            feature_importances = feature_importances.drop([col], axis =1)

    return feature_importances.columns




def identify_single_value_features(data): 
    ''' Identify features with single value 
    return a list of features '''  
    
    # Count unique values for each feature
    unique = data.nunique()
    unique = pd.DataFrame(unique).reset_index().rename(columns = {'index' : 'features', 0 :'nb_unique_values'})
    
    single_value_feature = []
    for i in range(len(unique)): 
        if unique.loc[i, 'nb_unique_values'] == 1 : 
            single_value_feature.append(unique.loc[i])
        
    return single_value_feature