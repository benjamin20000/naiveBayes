from model_io import read_model

class Classifier:
    """naive bayes classifier using a pre-trained model."""

    def __init__(self, model):
        """load model dict and score from file."""
        self.model_dict, self.model_score = read_model(model)
        self.model_likelihood_dict = self.model_dict["likelihood"]
        self.model_prior_dict = self.model_dict["priors"]

    def calculate_likelihood(self, data_point, target):
        """calculate likelihood of data_point for a given target."""
        likelihood = 1
        for feature in data_point.keys():
            value = data_point[feature]  # value in the data point
            if value in self.model_likelihood_dict[target][feature]:
                likelihood *= self.model_likelihood_dict[target][feature][value]
            else:
                likelihood *= self.model_likelihood_dict[target][feature]["value_feature_not_exist"]
        return likelihood

    def calculate_prediction(self, data_point):
        """predict target class with highest posterior probability."""
        scores = {}
        for target in self.model_likelihood_dict.keys():
            prior = self.model_prior_dict[target]
            likelihood = self.calculate_likelihood(data_point, target)
            scores[target] = prior * likelihood
        return max(scores, key=scores.get)

    def get_model_accuracy(self):
        """return model accuracy score."""
        return self.model_score
