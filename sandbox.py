model_options = {
    'Regressions': {
        'LogisticRegression': "LogisticRegression(random_state=random_state, solver='liblinear', max_iter=200)"
    },
    'Machine Learning': {
        'DecisionTreeClassifier': 'DecisionTreeClassifier(random_state=random_state)',
        'RandomForestClassifier': 'RandomForestClassifier(random_state=random_state)',
        
    }
}

for _, models in model_options.items():
    for model_name, model_code in models.items():
        print(f"{model_name}\n{model_code}")

transformed_data = ('asdf', 'asdf', 'asdf', 'asdf', 'asdf', 'asdf')

print(f'transformed_data: {[transformed_data]}')
    