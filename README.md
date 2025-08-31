# Data Toolbox
Common data science tasks as functions. Studying, amending, and analyzing data is an iterative process. It is normal to return to previous steps and correct/expand them to allow for further steps. Below are the methods in a general order.

## DataProcessor(df, random_state, target_name)
<p>.view(table, column, full_screen)
<p>.see(column, x_axis_name, top_values)
<p>.bootstrap(n_samples, n_rows, frac_rows, replace, weights, random_state, axis, print_head)
<p>.downsample(target_name, n_target_majority, n_rows, random_state, print_head)
<p>.upsample(target_name, n_target_majority, n_rows, random_state, print_head)
<p>.missing_values(column, missing_values_method, fill_value, print_head)
<p>.feature_scaler(column_names, print_head)
<p>.encode_features(model_type, ordinal_cols, categorical_cols, auto_encode, print_head)
<p>.split(split_ratio, target_name, random_state, print_head)
<p>.get_split(which, columns, print_head)
<p>.vectorize(df, features, target, print_head)
<p>.get_vectorized(which, columns, print_head)


## DataModeler(model_type, model_name, features, target, random_state, step_size, epochs, batch_size, reg_weight,  valid_features_vectorized, valid_target_vectorized, test_features, test_target)
<p>.fit(train_features_vectorized, train_target_vectorized, model_type, model_name, model_params, verbose)
<p>.predict(X, return_proba)
<p>.score(valid_features_vectorized, valid_target_vectorized, param_to_optimize, param_optimization_range, tolerance, max_iter, metric, manual_params, verbose)






