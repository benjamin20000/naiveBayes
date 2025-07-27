class Model:
    # splits the dataframe into train and test sets
    @staticmethod
    def split_train_test(df, train_percent=0.7):
        df_shuffled = df.sample(frac=1, random_state=67).reset_index(drop=True)  # shuffle the data
        split_idx = int(len(df_shuffled) * train_percent)  # compute split index
        train_df = df_shuffled.iloc[:split_idx]
        test_df = df_shuffled.iloc[split_idx:]
        return train_df, test_df  # returning a tuple of the split dataframes

    # creates a naive bayes model based on the training data
    @staticmethod
    def create_model(target_name, train_df):
        if target_name not in train_df:
            raise Exception(f"Target column '{target_name}' not found in the data")

        likelihood_data = {}
        priors = {}
        unique_targets = train_df[target_name].unique()  # get unique class labels

        for target in unique_targets:
            target_dict = {}
            relevant_data = train_df[train_df[target_name] == target]  # filter rows with current target

            for feature in relevant_data.columns:
                if feature == target_name:
                    continue
                value_counts = relevant_data[feature].value_counts().to_dict()  # count feature values
                k = train_df[feature].nunique()  # number of unique values in feature
                total = len(relevant_data)
                target_dict[feature] = {}

                for feature_value, count in value_counts.items():
                    # apply laplace smoothing
                    target_dict[feature][feature_value] = (count + 1) / (total + k)

                # probability for unseen feature values
                target_dict[feature]["value_feature_not_exist"] = 1 / (total + k)

            likelihood_data[target] = target_dict
            priors[target] = train_df[target_name].value_counts()[target] / len(train_df[target_name])

        model = {"priors": priors, "likelihood": likelihood_data}
        return model






