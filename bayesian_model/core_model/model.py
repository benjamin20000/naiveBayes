class model:
    @staticmethod
    def split_train_test(df, train_percent=0.7):
        """split dataframe into shuffled train and test sets."""
        df_shuffled = df.sample(frac=1, random_state=67).reset_index(drop=True)
        split_idx = int(len(df_shuffled) * train_percent)
        train_df = df_shuffled.iloc[:split_idx]
        test_df = df_shuffled.iloc[split_idx:]
        return (train_df, test_df)  # returning a tuple of the split df

    @staticmethod
    def create_model(target_name, train_df):
        """build naive bayes model with priors and likelihoods from training data."""
        if target_name not in train_df:
            raise Exception(f"Target column '{target_name}' not found in the data")

        likelihood_data = {}
        priors = {}
        unique_targets = train_df[target_name].unique()

        for target in unique_targets:
            target_dict = {}
            relevant_data = train_df[train_df[target_name] == target]

            for feature in relevant_data.columns:
                if feature == target_name:
                    continue
                value_counts = relevant_data[feature].value_counts().to_dict()
                k = train_df[feature].nunique()
                total = len(relevant_data)
                target_dict[feature] = {}
                for feature_value, count in value_counts.items():
                    target_dict[feature][feature_value] = (count + 1) / (total + k)
                target_dict[feature]["value_feature_not_exist"] = 1 / (total + k)

            likelihood_data[target] = target_dict
            priors[target] = train_df[target_name].value_counts()[target] / len(train_df[target_name])

        model = {"priors": priors, "likelihood": likelihood_data}
        return model
