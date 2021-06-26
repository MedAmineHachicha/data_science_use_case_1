from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from category_encoders.ordinal import OrdinalEncoder


class FeatureProcessor(object):
    """
    class to perform preprocessing on data using sklearn pipelines
    """
    def __init__(self, categ_cols, cont_cols):
        cat_encoder = OrdinalEncoder(handle_unknown='value')  # if a value is present in test but not in train, impute with -1
        cat_imputer = SimpleImputer(strategy="most_frequent")  # Replace missing values with most frequent category
        cont_imputer = SimpleImputer(strategy="mean")  # Replace missing values with mean

        self.preprocessor = ColumnTransformer(
            transformers=[('cont_cols', make_pipeline(cont_imputer, StandardScaler()), cont_cols),
                          ('categ_cols', make_pipeline(cat_imputer, cat_encoder, OneHotEncoder(drop='first')),
                           categ_cols)
                          ])

    def fit(self, X):
        """
        Fir the preprocessor on data
        :param X: pandas DataFrame containing categ_cols and cont_cols
        :return: self
        """
        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        """
        Transform data with fitted PreProcessor
        :param X: pandas DataFrame containing categ_cols and cont_cols
        :return: Transformed DataFrame as a numpy array
        """
        Transformed_data = self.preprocessor.transform(X)
        return Transformed_data