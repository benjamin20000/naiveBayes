from tqdm import tqdm

class tester:
    """test a naive bayes model on a labeled dataframe."""

    def __init__(self, df, model_dict, target_name):
        """initialize tester with data, model, and target column name."""
        self.df = df
        self.model_likelihood_dict = model_dict["likelihood"]
        self.model_prior_dict = model_dict["priors"]
        self.target_name = target_name

    def calculate_likelihood(self, data_point, target):
        """calculate likelihood of a data point for a given target."""
        likelihood = 1
        for feature in data_point.keys():
            value = data_point[feature]  # the value in the data point
            if value in self.model_likelihood_dict[target][feature]:
                likelihood *= self.model_likelihood_dict[target][feature][value]
            else:
                likelihood *= self.model_likelihood_dict[target][feature]["value_feature_not_exist"]
        return likelihood

    def calculate_bayes_score(self, data_point):
        """predict the most likely target class for a data point."""
        scores = {}
        for target in self.model_likelihood_dict.keys():
            prior = self.model_prior_dict[target]
            likelihood = self.calculate_likelihood(data_point, target)
            scores[target] = prior * likelihood
        print(scores)
        print(max(scores, key=scores.get))
        return max(scores, key=scores.get)

    def test(self):
        """test model predictions and return accuracy percentage."""
        success_count = 0
        total = len(self.df)
        for index, row in tqdm(self.df.iterrows(), total=total):
            data_point = row.drop(self.target_name).to_dict()
            predicted_target = self.calculate_bayes_score(data_point)
            print(f"{predicted_target}, {row[self.target_name]}")
            if predicted_target == row[self.target_name]:
                success_count += 1
        accuracy = (success_count / total) * 100
        print(success_count)
        print(total)
        return accuracy
