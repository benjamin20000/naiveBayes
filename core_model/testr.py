from tqdm import tqdm
from .classifier import Classifier


class Tester:
    def __init__(self,training_df, model_name, target_name):
        self.training_df = training_df
        self.df_len = len(training_df)
        self.target_name = target_name
        self.classifier = Classifier(model_name)

    def test(self):
        success_count = 0
        total = len(self.training_df)
        for index, row in tqdm(self.training_df.iterrows(), total=len(self.training_df)):
            data_point = row.drop(self.target_name).to_dict()
            predicted_target = self.classifier.calculate_prediction(data_point)
            if int(predicted_target) == row[self.target_name]:
                success_count += 1


        accuracy = (success_count / total) * 100
        return accuracy


