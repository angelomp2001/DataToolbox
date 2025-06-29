6/22/2025 Uploaded the start of my code

6/23/2025 created data_transformer.py, which handles all data transformation

6/24/2025 moved data_splitter() so it only runs once per model type.

6/25/2025 passed test_df, model_options, and existing optimal hyperparameters back through best_model_picker to test on test_df, but getting perfect scores.

6/26/2025 added test_features and test_target to best_model_picker, so how when you run it twice, it trains in training, but tests in testing data.

6/27/2025 refactored code hyperparameter_optimizer() optimizes both hyperparameters and threshold for regression models.  and categorical_scorer() only returns one iteration of scores.  so now, the hyperparameter_optimizer() primary role is the algo for optimizing the paramter and hosts the iterative algos for doing that.

6/28/2025 learned AI told me the wrong thing. Actually, every model needs threshold classification optimization. to do: ensure data transformation is once per model type, optimize thereshold for all models, output all scores in the end.  
