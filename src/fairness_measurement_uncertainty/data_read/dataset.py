from sklearn.model_selection import train_test_split


class Dataset():
    def __init__(self, test_features,
             test_features_drop_scores,
             test_labels,
             train_features,
             train_labels,
             train_scores,
             dev_features,
             dev_labels,
             dev_scores,
             ):
        self.test_features = test_features
        self.test_labels = test_labels
        self.train_features = train_features
        self.train_labels = train_labels
        self.train_scores = train_scores
        self.dev_features = dev_features
        self.dev_labels = dev_labels
        self.dev_scores = dev_scores
        self.test_features_drop_scores = test_features_drop_scores

    @staticmethod
    def from_full_data(X, y, test_size, score_field):
        """
        Splits the data into train, dev and test
        :param X: Input features
        :param y: Input lables
        :param test_size: fraction of data used for test
        :param score_field: Name of the ranking scores column
        :return: Dataset object containing the splits
        """

        # Sampling test data - test data is used for evaluating the correction schemes
        train_dev_features, test_features, train_dev_labels, test_labels = train_test_split(
            X, y, test_size=test_size)

        # Sampling development/train data - development data is used for estimating the class conditional error
        # Rates of the proxy model, train data is used for learning the proxy model.
        train_features, dev_features, train_labels, dev_labels = train_test_split(
            train_dev_features, train_dev_labels, test_size=test_size)

        train_scores = train_features[score_field]
        train_features = train_features.drop([score_field], axis=1)
        dev_scores = dev_features[score_field]
        dev_features = dev_features.drop([score_field], axis=1)
        test_features_drop_scores = test_features.drop([score_field], axis=1)
        return Dataset(test_features,
             test_features_drop_scores,
             test_labels,
             train_features,
             train_labels,
             train_scores,
             dev_features,
             dev_labels,
             dev_scores,

         )