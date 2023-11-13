from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy.stats import stats
from sklearn.metrics import confusion_matrix, accuracy_score

from data_read.real_dataset_builder import RealDatasetBuilder
from data_read.simulated_dataset_builder import SimulatedDatasetBuilder
from inclusive_search_fairness_measurement.exposure.wrapper import exposure_factory
from inclusive_search_fairness_measurement.pairwise_parity.wrapper import dp_factory
from label_predicting.flip_classifier import FlipModel
from label_predicting.lambdamart_ranker import LambdaMARTRanker
from label_predicting.logistic_regression_classifier import LRModel
from label_predicting.lstm_classifier import LSTMModel
from label_predicting.nn_classifier import NNModel
from label_predicting.rf_classifier import RFModel
from label_predicting.svm_classifier import SVMModel
from wrapper import rnd_factory
from experiments.experiments_stats import assumption_test


class Pipeline:
    def __init__(self, dataset_builder, metric, model, params: Dict):
        self.dataset_builder = dataset_builder
        self.metric = metric
        self.model = model
        self.params = params
        self.worldview = int(params["worldview"])
        self.model_name = params['model.name']
        self.metric_name = params['metric.name']

    @staticmethod
    def from_config(params: Dict) -> "Pipeline":
        """
        Instantiates a Pipeline object according to the given parameters
        :param params: A dictionary containing the necessary parameters
        """
        model_name = params["model.name"]
        metric_name = params["metric.name"]
        worldview = int(params["worldview"])

        # Prediction model-related param setting
        if model_name == "LR":
            model = LRModel()
        elif model_name == "flip":
            model = FlipModel.from_config(params)
        elif model_name == "SVM":
            model = SVMModel()
        elif model_name == "RF":
            model = RFModel()
        elif model_name == "MLP":
            model = NNModel()
        elif model_name == 'LSTM':
            model = LSTMModel()
        else:
            raise RuntimeError(f"unsupported model name: {model_name}")

        # Metric-related param settings
        if metric_name == "dp":  # Demographic parity
            metric = dp_factory(worldview)
        elif metric_name == "exposure":
            exposure_dropoff_str = params["metric.exposure.dropoff"]
            metric = exposure_factory(exposure_dropoff_str, worldview)
        elif metric_name == "rnd":
            ratio = params['metric.rnd.ratio']
            metric = rnd_factory(ratio, worldview)
        else:
            raise RuntimeError(f"unsupported metric name: {metric_name}")

        dataset_type = params["dataset.type"]
        if dataset_type == "real":
            dataset_builder = RealDatasetBuilder.dataset_builder_factory(params)
        elif dataset_type == 'sim':
            dataset_builder = SimulatedDatasetBuilder.from_config(params)
        else:
            raise RuntimeError(f"unsupported dataset type: {dataset_type}")

        return Pipeline(dataset_builder, metric, model, params)

    def run_experiment(self) -> dict:
        """
        Run the experiment given the settings and return
        :return: A dict containing the statistics of the experiment
        """

        dataset_type = self.params["dataset.type"]
        worldview = int(self.params["worldview"])

        # Gather data
        dataset_obj = self.dataset_builder.get_dataset()

        # estimating population demographics
        is_sensitive_train = np.array(dataset_obj.train_labels == self.dataset_builder.sens_att_value).astype(
            int
        )
        s_estimate = np.sum(is_sensitive_train) / is_sensitive_train.size

        # setting features for predicting sensitive attributes
        sensitive_pred_features = self.dataset_builder.sensitive_pred_features

        # predict labels in development and test data
        self.model.train(dataset_obj.train_features[sensitive_pred_features], dataset_obj.train_labels)
        if dataset_type == "sim" and worldview == 2:
            # no need to predict the estimated labels
            predicted_labels_dev = dataset_obj.dev_features["estimated_labels"]
            predicted_labels_test = dataset_obj.test_features_drop_scores["estimated_labels"]
        else:
            predicted_labels_dev = self.model.predict(dataset_obj.dev_features[sensitive_pred_features],
                                                      dataset_obj.dev_labels)
            predicted_labels_test = self.model.predict(dataset_obj.test_features_drop_scores[sensitive_pred_features],
                                                       dataset_obj.test_labels)

        # Update scores should they be predicted by LTR models
        if dataset_type == "real" and worldview == 2:
            score_pred_features = self.dataset_builder.score_pred_features
            # add A hat to the train
            predicted_labels_train = self.model.predict(dataset_obj.train_features[sensitive_pred_features],
                                                        dataset_obj.train_labels)
            score_train = pd.DataFrame(dataset_obj.train_features[score_pred_features])
            score_train["sensitive_hat"] = predicted_labels_train
            self.update_scores(dataset_obj, score_train, score_pred_features)

        # Compute actual fairness
        is_sensitive = np.array(dataset_obj.test_labels == self.dataset_builder.sens_att_value).astype(int)
        positions = -np.array(dataset_obj.test_features[self.dataset_builder.score_field])
        true_value = self.metric.compute(positions, is_sensitive, s_estimate)

        # computing the demographic parity according to predictions of the proxy model
        is_sensitive_dev = np.array(
            predicted_labels_dev == self.dataset_builder.sens_att_value
        ).astype(int)
        s_prime = np.sum(is_sensitive_dev) / is_sensitive_dev.size

        # 5. Compute the estimated fairness
        is_sensitive = np.array(
            predicted_labels_test == self.dataset_builder.sens_att_value
        ).astype(int)
        positions = -np.array(dataset_obj.test_features[self.dataset_builder.score_field])
        estimated_value = self.metric.compute(positions, is_sensitive, s_prime)
        corrected_value = estimated_value

        # 6. Parity correction
        conf_matrix = confusion_matrix(dataset_obj.dev_labels, predicted_labels_dev,
                                       labels=self.dataset_builder.att_values)
        tn, fp, fn, tp = conf_matrix.ravel()
        accuracy = accuracy_score(dataset_obj.dev_labels, predicted_labels_dev)

        # testing the assumption
        test_view_data = pd.DataFrame()
        test_view_data["A"] = pd.Series(dataset_obj.dev_labels).to_numpy()
        test_view_data["Ahat"] = pd.Series(predicted_labels_dev).to_numpy()
        test_view_data["S"] = pd.Series(dataset_obj.dev_scores).to_numpy()
        assumption_test_result = assumption_test(test_view_data, worldview, self.dataset_builder.sens_att_value)
        print(assumption_test_result)

        # computing empirical error rates
        emp_g1_error_rate = fn / (tp + fn)  # empirical q
        emp_g2_error_rate = fp / (tn + fp)  # empirical p
        print('error rates:', emp_g1_error_rate, emp_g2_error_rate)
        if self.params["model.name"] == "flip":
            sum_flip_error = self.params["model.flip.g1_error_rate"] + self.params["model.flip.g2_error_rate"]
        else:
            sum_flip_error = 0.0
        if float(emp_g1_error_rate + emp_g2_error_rate - 1.0) != 0.0 and sum_flip_error != 1.0:
            corrected_value = self.metric.correct(estimated_value, emp_g1_error_rate, emp_g2_error_rate, s_estimate)

        stat_values = {'test_res': assumption_test_result, 'model_name': self.model_name,
                       'g1_error': emp_g1_error_rate, 'g2_error': emp_g2_error_rate, 'accuracy': accuracy,
                       'true_value': true_value, 'estimated_value': estimated_value,
                       'corrected_value': corrected_value, 'metric_name': self.metric_name
                       }

        return stat_values

    def update_scores(self, dataset_obj, train_data, score_pred_features):
        """
        Trains a LTR model and updates the ranking scores using the predicted values
        :param dataset_obj: Input dataset object containing different splits
        :param train_data: Input features to the LTR model
        :param score_pred_features: Selected features for training
        """
        # train LTR model for predicting scores
        ltr_model = LambdaMARTRanker()
        ltr_model.train(train_data, dataset_obj.train_scores)

        # update dataset_obj.dev_scores
        y1 = ltr_model.predict(dataset_obj.dev_features[score_pred_features], None)
        # update dataset_obj.test_features[self.dataset_builder.score_field]
        y2 = ltr_model.predict(dataset_obj.test_features_drop_scores[score_pred_features], None)

        # compute Kendal tau
        tau, _ = stats.kendalltau(dataset_obj.dev_scores, y1)
        print(f'ranking tau {tau}')

        # update scores
        dataset_obj.dev_scores = y1
        dataset_obj.test_features[self.dataset_builder.score_field] = y2




