
# Create a new class

# In __init__:
# - List ALL parameters used in any of the related functions
# - Assign these parameters to instance variables (self.xxx)

# Copy each function into the class as a method

# Remove function parameters that are now instance variables

# Replace references to parameters with self.parameter_name

Class data_modeler:
    def __init__(self,
        self,


# .fit()
# .train(train= df, validate = df, test = df) if none or same provided, then X-validation
# .score(hyperparameters = , objective = 'optimize') if none, score, if provided, score, optimize if objective.
# .pred(features, target)