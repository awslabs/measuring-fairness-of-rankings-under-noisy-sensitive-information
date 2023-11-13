import random
import pandas as pd
import numpy as np
import re


from typing import List, Dict
from fairlearn.datasets import fetch_adult
from folktables import ACSDataSource
from langdetect import detect
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from data_read.dataset import Dataset
from label_predicting.feature_importance import feature_importance


class RealDatasetBuilder:
    def __init__(
            self,
            data_path: str,
            sensitive_att_name: str,
            sens_att_value_str: str,
            non_sens_att_value_str: str,
            sens_att_value: float,
            non_sens_att_value: float,
            score_field: str,
            att_values: List[int],
            test_size: float,
            target_bins: List[float]
    ):
        self.data_path = data_path

        self.sensitive_att_name = sensitive_att_name
        self.sens_att_value_str = sens_att_value_str
        self.non_sens_att_value_str = non_sens_att_value_str
        self.sens_att_value = sens_att_value
        self.non_sens_att_value = non_sens_att_value
        self.score_field = score_field
        self.att_values = att_values

        self.test_size = test_size

        # Binning the relevance score based on which the items are sorted (not used in our experiments)
        self.target_bins = target_bins

        self.selected_features = None
        self.cat_features = None
        self.full_dataset = None

        self.sensitive_pred_features = []
        self.score_pred_features = []

        self.lists = []

    def read_dataset(self):
        """
        Use self.data_path to read the full dataset from disk
        """
        raise NotImplementedError()

    # No longer used
    # def feature_selection(self):
    #     """
    #     Finds the features to remove for the first assumption to hold
    #     """
    #     assert self.full_dataset is not None, "Must read the data first"
    #
    #     X = self.full_dataset
    #     y = self.full_dataset[self.score_field]
    #     for feature in self.selected_features:
    #         if feature not in [self.score_field, self.sensitive_att_name]:
    #             sens_indices = X[self.sensitive_att_name] == self.sens_att_value_str
    #             non_sens_indices = X[self.sensitive_att_name] != self.sens_att_value_str
    #             v1 = X[sens_indices][feature]
    #             y1 = y[sens_indices]
    #             v2 = X[non_sens_indices][feature]
    #             y2 = y[non_sens_indices]
    #
    #             if feature not in self.cat_features:
    #                 from scipy.stats import spearmanr
    #                 corr1, _ = spearmanr(v1, y1)
    #                 corr2, _ = spearmanr(v2, y2)
    #                 avg_corr = (corr1 + corr2) / 2
    #                 print(f'feature {feature}: avg corr: {avg_corr}')
    #             else:
    #                 X1 = X[sens_indices]
    #                 a = X1[[feature, self.score_field]].groupby(feature, as_index=False).mean()
    #                 variance = np.var(a[self.score_field])
    #                 print(f'feature {feature}, var: {variance}')
    #     removed_features = ["workclass", "education", "education-num", "age", "relationship", "native-country",
    #                         "target", "occupation"]
    #     self.selected_features = list(self.selected_features)
    #     for f in removed_features:
    #         self.selected_features.remove(f)

    # No longer used
    # def feature_selection_importance(self, model, n_features, n_exclude):
    #     """
    #     Use the model to generate the list of relevant features. Must read the data first
    #     """
    #     assert self.full_dataset is not None, "Must read the data first"
    #
    #     X = self.full_dataset.drop([self.sensitive_att_name, self.score_field], axis=1)
    #
    #     if model is None:
    #         self.selected_features = list(self.full_dataset.columns.values)
    #         return
    #
    #     # # TODO to be removed
    #     # from sklearn.neural_network import MLPClassifier
    #     # from sklearn.ensemble import RandomForestClassifier
    #     # from sklearn import svm
    #     # from sklearn.linear_model import LogisticRegression
    #     # if type(model) == MLPClassifier:
    #     #     self.selected_features = ['marital-status', 'race', 'hours-per-week', 'sex']
    #     # elif type(model) == RandomForestClassifier:
    #     #     self.selected_features = ['marital-status', 'workclass', 'hours-per-week', 'sex']
    #     # elif type(model) == svm.LinearSVC:
    #     #     self.selected_features = ['marital-status', 'fnlwgt', 'hours-per-week', 'sex']
    #     # elif type(model) == LogisticRegression:
    #     #     self.selected_features = ['marital-status', 'native-country', 'hours-per-week', 'sex']
    #     # else:
    #     #     raise ValueError(f'unsupported type of the model: {type(model)}')
    #     # return
    #
    #     # TODO remove the following line
    #     # self.selected_features = list(self.full_dataset.columns.values)
    #     # return
    #
    #     # feature importance for predicting relevance score
    #     y = self.full_dataset[self.score_field]
    #     if self.target_bins is not None:
    #         labels = [str(i) for i in self.target_bins]
    #         y = pd.cut(self.full_dataset[self.score_field], self.target_bins, labels=labels[:-1])
    #     score_importance_values = feature_importance(copy.deepcopy(model), X, self.cat_features, y)
    #     sorted_features = sorted(score_importance_values.items(), key=lambda x: x[1], reverse=True)
    #     print(sorted_features)
    #     score_sorted_features = [e[0] for e in sorted_features]
    #
    #     # feature importance for predicting sensitive attribute
    #     y = self.full_dataset[self.sensitive_att_name]
    #     sens_att_importance_values = feature_importance(copy.deepcopy(model), X, self.cat_features, y)
    #     sorted_features = sorted(sens_att_importance_values.items(), key=lambda x: x[1], reverse=True)
    #     print(sorted_features)
    #     sens_att_sorted_features = [e[0] for e in sorted_features]
    #
    #     # selection using n_features, n_exclude
    #     selected_features = []
    #     for feature in sens_att_sorted_features:
    #         if len(selected_features) < n_features:
    #             if feature not in score_sorted_features[:n_exclude]:
    #                 selected_features.append(feature)
    #
    #     if len(selected_features) != n_features:
    #         raise RuntimeError("Feature selection failed to continue, could not find enough features")
    #
    #     # TODO consider different methods for selectnig the final set of features
    #     # feature_score = alpha * (1 / rank in sens_att_sorted_features) + (1 / inverse rank in score_sorted_features)
    #
    #     selected_features.append(self.score_field)
    #     selected_features.append(self.sensitive_att_name)
    #
    #     self.selected_features = selected_features

    # No longer used
    # def overlapping_feature_selection(self, model, n_features, n_range):
    #     """
    #     Use the model to generate the list of features that are important for both sens_att and score
    #     """
    #     assert self.full_dataset is not None, "Must read the data first"
    #
    #     X = self.full_dataset.drop([self.sensitive_att_name, self.score_field], axis=1)
    #
    #     if model is None:
    #         self.selected_features = list(self.full_dataset.columns.values)
    #         return
    #
    #     # feature importance for predicting relevance score
    #     y = self.full_dataset[self.score_field]
    #     if self.target_bins is not None:
    #         labels = [str(i) for i in self.target_bins]
    #         y = pd.cut(self.full_dataset[self.score_field], self.target_bins, labels=labels[:-1])
    #     score_importance_values = feature_importance(copy.deepcopy(model), X, self.cat_features, y)
    #     sorted_features = sorted(score_importance_values.items(), key=lambda x: x[1], reverse=True)
    #     print(sorted_features)
    #     score_sorted_features = [e[0] for e in sorted_features]
    #
    #     # feature importance for predicting sensitive attribute
    #     y = self.full_dataset[self.sensitive_att_name]
    #     sens_att_importance_values = feature_importance(copy.deepcopy(model), X, self.cat_features, y)
    #     sorted_features = sorted(sens_att_importance_values.items(), key=lambda x: x[1], reverse=True)
    #     print(sorted_features)
    #     sens_att_sorted_features = [e[0] for e in sorted_features]
    #
    #     # selection using n_features, n_range
    #     selected_features = []
    #     for feature in sens_att_sorted_features:
    #         if len(selected_features) < n_features:
    #             if feature in score_sorted_features[:n_range]:
    #                 selected_features.append(feature)
    #
    #     if len(selected_features) != n_features:
    #         raise RuntimeError("Feature selection failed to continue, could not find enough features")
    #
    #     selected_features.append(self.score_field)
    #     selected_features.append(self.sensitive_att_name)
    #
    #     self.selected_features = selected_features

    def get_processed_dataset(self):
        # Implements the necessary preprocessing

        raise NotImplementedError()

    def get_dataset(self):
        """
        Returns the working version of the dataset
        """

        # Get the processed data
        features = self.get_processed_dataset()
        X = features.drop([self.sensitive_att_name], axis=1)
        y = features[self.sensitive_att_name]
        if type(y[0]) == str:
            y = pd.Series(
                np.where(y.str.contains(self.sens_att_value_str), self.sens_att_value, self.non_sens_att_value))

        # MinMax scaling
        if len(X._get_numeric_data().columns) > 1:
            scaler = MinMaxScaler()
            X = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

        dataset_obj = Dataset.from_full_data(X, y, self.test_size, self.score_field)

        return dataset_obj

    def set_predictive_features(self):
        """
        Sets the features used for predicting the sensitive attribute and the ranking scores (for Assumption II)
        :return:
        """
        self.sensitive_pred_features = self.selected_features
        self.score_pred_features = self.selected_features

    @staticmethod
    def dataset_builder_factory(real_params: Dict) -> "RealDatasetBuilder":
        """
        Returns the dataset-specific builder

        :param real_params: Parameters read from the config file
        :return: Return the correct datasetbulder based on the given dataset
        """

        dataset = real_params["dataset.name"]
        sensitive_att_name = real_params["dataset.sensitive_att_name"]

        str_val_map = real_params["dataset.att_values_numeric"]
        sens_att_value_str = real_params["dataset.sens_att_value"]
        sens_att_value = int(str_val_map[sens_att_value_str])
        non_sens_att_value_str = real_params["dataset.non_sens_att_value"]
        non_sens_att_value = int(str_val_map[non_sens_att_value_str])

        score_field = real_params["dataset.score_field"]
        att_values = [non_sens_att_value, sens_att_value]
        test_size = float(real_params["dataset.test_size"])
        data_path = real_params["dataset.path"]

        args = [data_path,
            sensitive_att_name,
            sens_att_value_str,
            non_sens_att_value_str,
            sens_att_value,
            non_sens_att_value,
            score_field,
            att_values,
            test_size]

        # target_bins is no longer used
        if dataset == "adult":
            target_bins = [-2, 40, 80, 100]
            args.append(target_bins)
            return RealAdultDatasetBuilder(*args)
        elif dataset == "compas":
            target_bins = None
            args.append(target_bins)
            return RealCompasDatasetBuilder(*args)
        elif dataset == 'fifa':
            target_bins = None
            args.append(target_bins)
            return RealFIFADatasetBuilder(*args)
        elif dataset == 'usaname':
            target_bins = None
            args.append(target_bins)
            return RealUSANamesDatasetBuilder(*args)
        elif dataset == 'goodreads':
            target_bins = None
            args.append(target_bins)
            return RealGoodreadsAuthorsDatasetBuilder(*args)
        elif dataset == 'artists':
            target_bins = None
            args.append(target_bins)
            return RealArtistsDatasetBuilder(*args)
        elif dataset == 'hr':
            target_bins = None
            args.append(target_bins)
            return RealHRDatasetBuilder(*args)
        elif dataset == 'exam':
            target_bins = [-1, 2, 3, 4]
            args.append(target_bins)
            return RealExamDatasetBuilder(*args)
        elif dataset == 'tommy':
            target_bins = None
            args.append(target_bins)
            return RealTommyDatasetBuilder(*args)
        else:
            raise RuntimeError(f"Unsupported dataset name: {dataset}")


class RealAdultDatasetBuilder(RealDatasetBuilder):
    def read_dataset(self):
        """
        Reads the raw adult dataset and set the list of categorical values
        """

        # ********* old adult dataset ***********
        # param setting
        self.cat_features = ['education', 'marital-status', 'occupation', 'relationship',
                        'race', 'native-country', 'workclass', 'target']
        # read dataset
        data = fetch_adult(cache=True, data_home=self.data_path, as_frame=True)
        self.full_dataset = data.data
        self.full_dataset['target'] = data.target.values

        self.selected_features = self.full_dataset.columns.values

        # ********* new adult dataset ***********
        # data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
        # acs_data = data_source.get_data(states=['TX'])
        # features = [
        #     'AGEP',
        #     'SCHL',
        #     'MAR',
        #     'RELP',
        #     'COW',
        #     'OCCP',
        #     'SEX',
        #     'RAC1P',
        #     'WKHP',
        #     'POBP',
        #     'PINCP'
        # ]
        # acs_data = acs_data[features]
        # acs_data.fillna(-1, inplace=True)
        # self.full_dataset = acs_data
        # self.selected_features = self.full_dataset.columns.values
        # self.cat_features = []

    def get_processed_dataset(self):
        """
        Return the preprocessed Adult UCI dataset
        """

        assert self.full_dataset is not None, "Must read the data first"

        # Data size: 48842 rows x 15 columns
        # data features: 'feature_names': ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
        # 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        # 'hours-per-week', 'native-country'], 'target_names': ['class']

        df = self.full_dataset[self.selected_features]
        cat_columns = list(set(self.cat_features) & set(self.selected_features))

        # One-hot encoding
        for column in cat_columns:
            tempdf = pd.get_dummies(df[column], prefix=column)
            df = pd.merge(
                left=df,
                right=tempdf,
                left_index=True,
                right_index=True,
            )
            df = df.drop(columns=column)
        print(df)
        # old code
        # if cat_columns:
        #     labelencoder = LabelEncoder()
        #     df2 = df[cat_columns].apply(lambda col: labelencoder.fit_transform(col))
        #     enc = OneHotEncoder(handle_unknown='ignore')
        #     enc_df = pd.DataFrame(enc.fit_transform(df2[cat_columns]).toarray())
        #     # merge with main df bridge_df on key values
        #     df_join = df.join(enc_df)
        #     df_join = df_join.drop(cat_columns, axis=1)
        #     df = df_join

        # Ordinal encoding
        # for feature in cat_columns:
        #     tmp = df[feature].mask(df[feature] == ' ').factorize()[0]
        #     df[feature] = tmp

        return df


class RealCompasDatasetBuilder(RealDatasetBuilder):
    def read_dataset(self):
        """
        Reads the COMPAS dataset and set the the categorical features
        Reference: https://github.com/propublica/compas-analysis
        """

        # param setting
        # all_features = ["Sex_Code_Text", "Ethnic_Code_Text", "DateOfBirth", "LegalStatus", "CustodyStatus", "MaritalStatus",
        # "RecSupervisionLevel", "Scale_ID", "DisplayText", "ScoreText"]
        all_features = ["Sex_Code_Text", "Ethnic_Code_Text", "DateOfBirth", "LegalStatus", "CustodyStatus",
                        "MaritalStatus", "ScoreText"]

        self.cat_features = ["Sex_Code_Text", "LegalStatus", "CustodyStatus", "MaritalStatus"]

        # read dataset
        df = pd.read_csv(self.data_path)
        df = df[all_features]
        now = pd.Timestamp('now')
        df['DateOfBirth'] = pd.to_datetime(df['DateOfBirth'], format='%m/%d/%y')
        df['DateOfBirth'] = df['DateOfBirth'].where(df['DateOfBirth'] < now, df['DateOfBirth'] - np.timedelta64(100, 'Y'))
        df['DateOfBirth'] = (now - df['DateOfBirth']).astype('<m8[Y]')
        df['Ethnic_Code_Text'] = pd.Series(np.where(df['Ethnic_Code_Text'].str.contains("African-American"),
                                                    "African-American", "Other"))
        df['ScoreText'] = df['ScoreText'].map({'Low': 3, 'High': 1, 'Medium': 2})
        df.fillna(-1, inplace=True)
        self.full_dataset = df
        self.selected_features = self.full_dataset.columns.values

    def get_processed_dataset(self):
        """
        Returns the preprocessed COMPAS dataset
        """

        assert self.full_dataset is not None, "Must read the data first"

        df = self.full_dataset[self.selected_features]
        cat_columns = list(set(self.cat_features) & set(self.selected_features))

        # One-hot encoding
        for column in cat_columns:
            tempdf = pd.get_dummies(df[column], prefix=column)
            df = pd.merge(
                left=df,
                right=tempdf,
                left_index=True,
                right_index=True,
            )
            df = df.drop(columns=column)
        # print(df)
        # old code
        # if cat_columns:
        #     labelencoder = LabelEncoder()
        #     df2 = df[cat_columns].apply(lambda col: labelencoder.fit_transform(col))
        #     enc = OneHotEncoder(handle_unknown='ignore')
        #     enc_df = pd.DataFrame(enc.fit_transform(df2[cat_columns]).toarray())
        #     # merge with main df bridge_df on key values
        #     df_join = df.join(enc_df)
        #     df_join = df_join.drop(cat_columns, axis=1)
        #     df = df_join

        # Ordinal encoding
        # for feature in cat_columns:
        #     tmp = df[feature].mask(df[feature] == ' ').factorize()[0]
        #     df[feature] = tmp

        return df

    def set_predictive_features(self):
        """
        Sets the features for for predicting the sensitive attribute and the ranking scores
        """
        all_features = ['DateOfBirth', 'LegalStatus_Pretrial', 'CustodyStatus_Jail Inmate',
                        'CustodyStatus_Probation', 'MaritalStatus_Single', 'LegalStatus_Post Sentence',
                        'MaritalStatus_Married', 'Sex_Code_Text_Male',
                        'CustodyStatus_Pretrial Defendant', 'MaritalStatus_Divorced', 'LegalStatus_Other',
                        'Sex_Code_Text_Female', 'MaritalStatus_Significant Other',
                        'MaritalStatus_Separated', 'LegalStatus_Conditional Release',
                        'MaritalStatus_Widowed', 'LegalStatus_Probation Violator', 'MaritalStatus_Unknown',
                        'CustodyStatus_Residential Program', 'CustodyStatus_Prison Inmate',
                        'LegalStatus_Deferred Sentencing', 'CustodyStatus_Parole', 'LegalStatus_Parole Violator']

        # self.sensitive_pred_features = ['DateOfBirth', 'LegalStatus_Pretrial', 'CustodyStatus_Jail Inmate',
        # 'CustodyStatus_Probation', 'MaritalStatus_Single', 'LegalStatus_Post Sentence', 'MaritalStatus_Married',
        # 'Sex_Code_Text_Male', 'CustodyStatus_Pretrial Defendant', 'MaritalStatus_Divorced']
        # self.score_pred_features = [
        #                             'CustodyStatus_Residential Program', 'CustodyStatus_Prison Inmate',
        #                             'LegalStatus_Deferred Sentencing', 'CustodyStatus_Parole',
        #                             'LegalStatus_Parole Violator']

        self.sensitive_pred_features = all_features[:12]
        self.score_pred_features = all_features[12:]


class RealFIFADatasetBuilder(RealDatasetBuilder):
    def read_dataset(self):
        """
        Returns FIFA dataset
        Reference: https://www.kaggle.com/stefanoleone992/fifa-20-complete-player-dataset
        """

        dataORIGINAL = pd.read_csv(self.data_path, header=0)
        dataORIGINAL = dataORIGINAL.loc[
            (dataORIGINAL['nationality'] == 'England') | (dataORIGINAL['nationality'] == 'Germany')].reset_index()

        self.cat_features = []
        dataORIGINAL = dataORIGINAL.rename(columns={"long_name": "input_string"})
        self.full_dataset = dataORIGINAL
        print(self.full_dataset.shape)
        self.selected_features = ["input_string"]

    def get_processed_dataset(self):
        """
        Return the processed FIFA dataset
        """

        assert self.full_dataset is not None, "Must read the data first"

        N = self.full_dataset.shape[0]
        full_names = self.full_dataset['input_string'].copy()

        # make sure that names have the same length
        name_lengths = [len(full_names.iloc[ell]) for ell in range(N)]
        max_name_length = max(name_lengths)
        for ell in range(N):
            if len(full_names.iloc[ell]) < max_name_length:
                full_names.iloc[ell] = full_names.iloc[ell] + str('?' * (max_name_length - name_lengths[ell]))

        df = pd.DataFrame()
        df['input_string'] = full_names
        df[self.sensitive_att_name] = self.full_dataset[self.sensitive_att_name]
        df[self.score_field] = self.full_dataset[self.score_field]

        return df


class RealUSANamesDatasetBuilder(RealDatasetBuilder):
    def read_dataset(self):
        """
        Gets the USA name dataset 1910-current
        Reference: https://www.kaggle.com/datagov/usa-names?select=usa_1910_current
        """
        df = pd.read_csv(self.data_path)
        df = df.loc[(1980 <= df['year']) & (df['year'] <= 1995)].reset_index()
        df = df[['name', self.sensitive_att_name, 'number']]
        df = df.groupby(['name']).agg(lambda x: x.value_counts().index[0]).reset_index()
        print(df)
        print(f'USA name dataset filtered data size: {df.shape[0]}')
        positions = df['name'].argsort()
        scores = -1 * positions
        df[self.score_field] = scores

        df = df.rename(columns={"name": "input_string"})
        all_features = ['input_string', self.score_field, self.sensitive_att_name]
        self.selected_features = all_features
        self.cat_features = []
        self.full_dataset = df

    def get_processed_dataset(self):
        """
        Returns the processed USA name dataset
        """

        assert self.full_dataset is not None, "Must read the data first"

        N = self.full_dataset.shape[0]
        names = self.full_dataset['input_string'].copy()

        # make sure that names have the same length
        name_lengths = [len(names.iloc[ell]) for ell in range(N)]
        max_name_length = max(name_lengths)
        for ell in range(N):
            if len(names.iloc[ell]) < max_name_length:
                names.iloc[ell] = names.iloc[ell] + str('?' * (max_name_length - name_lengths[ell]))

        df = pd.DataFrame()
        df['input_string'] = names
        df[self.sensitive_att_name] = self.full_dataset[self.sensitive_att_name]
        df[self.score_field] = self.full_dataset[self.score_field]

        return df


class RealGoodreadsAuthorsDatasetBuilder(RealDatasetBuilder):
    def read_dataset(self):
        """
        Gets the Goodreads Authors dataset
        Reference: https://www.kaggle.com/choobani/goodread-authors
        """
        df = pd.read_csv(self.data_path)
        all_features = ['name', 'workcount', 'fan_count', 'gender', 'average_rate', 'rating_count', 'review_count']
        self.cat_features = []
        df = df[all_features]
        positions = df['name'].argsort()
        scores = -1 * positions
        df["scores"] = scores
        df = df.rename(columns={"name": "input_string"})
        self.full_dataset = df
        self.selected_features = ["input_string"]

    def get_processed_dataset(self):
        """
        Returns the processed Goodreads Authors dataset
        """

        assert self.full_dataset is not None, "Must read the data first"

        N = self.full_dataset.shape[0]
        names = self.full_dataset['input_string'].copy()

        # Extract first names
        for ell in range(N):
            names.iloc[ell] = names.iloc[ell].split(' ')[0]

        # Make sure that names have the same length
        name_lengths = [len(names.iloc[ell]) for ell in range(N)]
        max_name_length = max(name_lengths)
        for ell in range(N):
            if len(names.iloc[ell]) < max_name_length:
                names.iloc[ell] = names.iloc[ell] + str('?' * (max_name_length - name_lengths[ell]))

        df = pd.DataFrame()
        df['input_string'] = names
        df[self.sensitive_att_name] = self.full_dataset[self.sensitive_att_name]
        df[self.score_field] = self.full_dataset[self.score_field]

        return df


class RealExamDatasetBuilder(RealDatasetBuilder):
    def read_dataset(self):
        """
        Read the UFRGS Entrance Exam and GPA Dataset
        Reference: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/O35FW8
        """
        # read dataset
        df = pd.read_csv(self.data_path, delimiter=';')
        df.fillna(-1, inplace=True)
        self.cat_features = []
        self.full_dataset = df
        self.selected_features = self.full_dataset.columns.values

    def get_processed_dataset(self):
        """
        Returns the preprocessed compass dataset
        """

        assert self.full_dataset is not None, "Must read the data first"

        df = self.full_dataset[self.selected_features]
        return df


class RealArtistsDatasetBuilder(RealDatasetBuilder):
    def read_dataset(self):
        """
        Reads the music artists popularity dataset
        https://www.kaggle.com/pieca111/music-artists-popularity
        """
        # Read dataset
        g1 = self.sens_att_value_str
        g2 = self.non_sens_att_value_str
        df = pd.read_csv(self.data_path, delimiter=',')
        df = df[(df['country_mb'].notnull()) & (df['artist_mb'].notnull()) &
                (df['listeners_lastfm'].notnull()) & (df['tags_mb'].notnull()) & (~df['ambiguous_artist'])]
        df = df[df['artist_mb'].str.match("[a-zA-Z]+")]
        df = df[(df.country_mb == g1) | (df.country_mb == g2)]
        # df = df[df['tags_mb'].str.contains(";")]

        df1 = df[df.country_mb == g1].sample(5000)
        # df2 = df[~df.index.isin(df1.index)].sample(1000)
        df2 = df[df.country_mb == g2].sample(5000)
        df = pd.concat([df1, df2]).reset_index()
        print(f'Artist dataset size: {df.shape}')
        # print(f'Artist dataset size {g1}: {df[df.country_mb == g1].shape}')
        self.cat_features = []
        self.full_dataset = df
        self.selected_features = self.full_dataset.columns.values

    def get_processed_dataset(self):
        """
        Return the processed FIFA dataset
        """

        assert self.full_dataset is not None, "Must read the data first"

        N = self.full_dataset.shape[0]
        full_names = self.full_dataset['artist_mb'].copy()

        # make sure that names have the same length
        name_lengths = [len(full_names.iloc[ell]) for ell in range(N)]
        max_name_length = max(name_lengths)
        for ell in range(N):
            if len(full_names.iloc[ell]) < max_name_length:
                full_names.iloc[ell] = full_names.iloc[ell] + str('?' * (max_name_length - name_lengths[ell]))

        df = pd.DataFrame()
        df['input_string'] = full_names
        df[self.sensitive_att_name] = self.full_dataset[self.sensitive_att_name]
        df[self.score_field] = self.full_dataset[self.score_field]

        return df


class RealHRDatasetBuilder(RealDatasetBuilder):
    def read_dataset(self):
        """
        Read the IBM HR Analytics Employee Attrition & Performance dataset and set the the categorical features
        Reference: https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset
        """

        # param setting
        all_features = ["Age", "Attrition", "BusinessTravel", "DailyRate",
                        "Department", "DistanceFromHome", "Education", "EducationField",
                        "EnvironmentSatisfaction", "Gender",
                        "HourlyRate", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction",
                        "MaritalStatus", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
                        "Over18", "OverTime", "PercentSalaryHike",
                        "PerformanceRating", "RelationshipSatisfaction", "DailyRate",
                        "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance",
                        "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"]

        self.cat_features = ["BusinessTravel", "Department", "EducationField", "JobRole", "Gender", "Over18",
                             "OverTime"]

        # Read dataset
        df = pd.read_csv(self.data_path)
        df = df[all_features]
        df.fillna(-1, inplace=True)
        df.loc[df.Attrition == 'Yes', "Attrition"] = 0
        df.loc[df.Attrition == 'No', "Attrition"] = 1
        df.loc[df.MaritalStatus != 'Single', "MaritalStatus"] = "Other"
        # print(df)
        self.full_dataset = df
        self.selected_features = self.full_dataset.columns.values

    def get_processed_dataset(self):
        """
        Return the preprocessed compass dataset
        """

        assert self.full_dataset is not None, "Must read the data first"

        df = self.full_dataset[self.selected_features]
        cat_columns = list(set(self.cat_features) & set(self.selected_features))

        # One-hot encoding
        for column in cat_columns:
            tempdf = pd.get_dummies(df[column], prefix=column)
            df = pd.merge(
                left=df,
                right=tempdf,
                left_index=True,
                right_index=True,
            )
            df = df.drop(columns=column)
        # print(df.columns.values)
        # old code
        # if cat_columns:
        #     labelencoder = LabelEncoder()
        #     df2 = df[cat_columns].apply(lambda col: labelencoder.fit_transform(col))
        #     enc = OneHotEncoder(handle_unknown='ignore')
        #     enc_df = pd.DataFrame(enc.fit_transform(df2[cat_columns]).toarray())
        #     # merge with main df bridge_df on key values
        #     df_join = df.join(enc_df)
        #     df_join = df_join.drop(cat_columns, axis=1)
        #     df = df_join

        # Ordinal encoding
        # for feature in cat_columns:
        #     tmp = df[feature].mask(df[feature] == ' ').factorize()[0]
        #     df[feature] = tmp

        return df

    def set_predictive_features(self):
        # self.score_pred_features = ['YearsAtCompany', 'YearsSinceLastPromotion', 'EnvironmentSatisfaction',
        #                             'JobRole_Research Scientist', 'DailyRate']

        all_features = ['Age', 'Attrition', 'MonthlyIncome', 'DistanceFromHome',
                        'HourlyRate', 'MonthlyRate', 'YearsWithCurrManager', 'TotalWorkingYears',
                        'JobSatisfaction', 'NumCompaniesWorked', 'EnvironmentSatisfaction',
                        'DailyRate', 'YearsSinceLastPromotion', 'PercentSalaryHike',
                        'YearsAtCompany', 'JobInvolvement', 'Education', 'JobLevel',
                        'PerformanceRating', 'RelationshipSatisfaction', 'TrainingTimesLastYear',
                        'WorkLifeBalance', 'BusinessTravel_Non-Travel',
                        'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
                        'Department_Human Resources', 'Department_Research & Development',
                        'Department_Sales', 'EducationField_Human Resources',
                        'EducationField_Life Sciences', 'EducationField_Marketing',
                        'EducationField_Medical', 'EducationField_Other',
                        'EducationField_Technical Degree',
                        'JobRole_Healthcare Representative', 'JobRole_Human Resources',
                        'JobRole_Laboratory Technician', 'JobRole_Manager',
                        'JobRole_Manufacturing Director', 'JobRole_Research Director',
                        'JobRole_Research Scientist', 'JobRole_Sales Executive',
                        'JobRole_Sales Representative', 'Gender_Female', 'Gender_Male',
                        'Over18_Y', 'OverTime_No', 'OverTime_Yes']

        self.sensitive_pred_features = all_features[:25]
        self.score_pred_features = all_features[25:]


class RealTommyDatasetBuilder(RealDatasetBuilder):
    def read_dataset(self):
        """
        Gets the Tommy dataset
        https://s3.console.aws.amazon.com/s3/object/alster-pii-workspaces?region=us-west-2&prefix=gazin/
        evaluating-bias-estimation-correction/part-00000-9b8ba1cf-77e0-4a57-9fd8-0bee7554acea-c000.json
        """

        # Read data
        df = pd.read_json(self.data_path, orient="records", lines=True)
        unique_names = {}
        for index, row in df.iterrows():
            ranking = {"positions": [], "names": [], "labels": []}
            for elem in row["list_metric_data"]:
                position = elem["organic_position"]
                name = elem["item_name"]
                labels = elem["attribute_values"]
                if len(labels) == 1:
                    # add to the ranking
                    ranking["positions"].append(position)
                    ranking["names"].append(name)
                    att_map = {"Women": 1, "Men": 0, "Boys": 0, "Girls": 1}
                    ranking["labels"].append(att_map[labels[0]])
                    unique_names[name] = att_map[labels[0]]
            self.lists.append(dict(ranking))
        print("datasetbuilder: len of lists", len(self.lists))

        # df of names
        names_df = pd.DataFrame(unique_names.items(), columns=['input_string', self.sensitive_att_name])
        names_df[self.score_field] = np.zeros(len(unique_names))

        self.cat_features = []
        self.full_dataset = names_df
        self.selected_features = ["input_string"]

    def get_processed_dataset(self):
        """
        Returns the processed Goodreads Authors dataset
        """

        assert self.full_dataset is not None, "Must read the data first"

        N = self.full_dataset.shape[0]
        names = self.full_dataset['input_string'].copy()

        # remove irrelevant characters
        regex = re.compile('[^a-zA-Z ]')
        for ell in range(N):
            names.iloc[ell] = regex.sub('', names.iloc[ell])

        # make sure that sentences have the same number of words
        name_lengths = [len(names.iloc[ell].split(' ')) for ell in range(N)]
        max_name_length = max(name_lengths)
        print(f"in dataset builder, max name legth {max_name_length}")
        for ell in range(N):
            if len(names.iloc[ell].split(' ')) < max_name_length:
                names.iloc[ell] = names.iloc[ell] + str(' ?' * (max_name_length - name_lengths[ell]))

        # update names in the lists
        for i in range(len(self.lists)):
            for j in range(len(self.lists[i]["names"])):
                name_length = len(self.lists[i]["names"][j].split(' '))
                self.lists[i]["names"][j] = regex.sub('', self.lists[i]["names"][j])
                if name_length < max_name_length:
                    self.lists[i]["names"][j] = self.lists[i]["names"][j] + str(' ?' * (max_name_length - name_length))

        df = pd.DataFrame()
        df['input_string'] = names
        df[self.sensitive_att_name] = self.full_dataset[self.sensitive_att_name]
        df[self.score_field] = self.full_dataset[self.score_field]

        return df

    def get_lists(self, fraction: float):
        # return self.lists[: int(fraction * len(self.lists))]
        return random.sample(self.lists, int(fraction * len(self.lists)))


if __name__ == "__main__":
    data_path = '/Users/gazin/Documents/Amazon-internship/amazon-internship/uncertainty-experiments/final_dataset.csv'

    # filtering the authors to the subset with english names
    # df = pd.read_csv(data_path)
    # print(f'size before deletion: {df.shape[0]}')
    #
    # def is_en(txt):
    #     try:
    #         return detect(txt) == 'en'
    #     except:
    #         return False
    #
    # df = df[df.name.apply(is_en)].reset_index()
    # df = df[df.gender != 'unknown'].reset_index()
    #
    # df.to_csv('/Users/gazin/Documents/Amazon-internship/amazon-internship/uncertainty-experiments/final_dataset_filtered.csv')