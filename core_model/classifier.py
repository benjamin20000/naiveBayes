from model_io import read_model

class Classifier:
    """
    Naive Bayes classifier using pre-trained model data.
    """

    def __init__(self, model_name):
        """
        Load model priors and likelihoods.
        """
        self.model_dict, self.model_score = read_model(model_name)
        self.model_likelihood_dict = self.model_dict["likelihood"]
        self.model_prior_dict = self.model_dict["priors"]

    def calculate_likelihood(self, data_point, target):
        """
        Compute likelihood of data_point given a target class.
        """
        likelihood = 1
        for feature in data_point.keys():
            value_str = str(data_point[feature])
            if value_str in self.model_likelihood_dict[target][feature]:
                likelihood *= self.model_likelihood_dict[target][feature][value_str]
            else:
                likelihood *= self.model_likelihood_dict[target][feature]["value_feature_not_exist"]
        return likelihood

    def calculate_prediction(self, data_point):
        """
        Predict class for a given data_point.
        """
        scores = {}
        for target in self.model_likelihood_dict.keys():
            prior = self.model_prior_dict[target]
            likelihood = self.calculate_likelihood(data_point, target)
            scores[target] = prior * likelihood
        return max(scores, key=scores.get)

    def get_model_accuracy(self):
        """
        Return model accuracy.
        """
        return self.model_score
