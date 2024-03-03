def Logistic_ML(df):
    #Logistic regression
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report

    # Split data into features (X) and target variable (y)
    X = df.drop('converted', axis=1)  # Features
    y = df['converted']  # Target variable

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize logistic regression model
    logreg = LogisticRegression()

    # Fit the model on the training data
    logreg.fit(X_train, y_train)

    # Predict on the test data
    y_pred = logreg.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Print classification report
    print(classification_report(y_test, y_pred))

    # Get coefficients
    coefficients = logreg.coef_[0]

    # Map coefficients to their corresponding feature names
    feature_coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': coefficients})

    # Sort the features by their coefficients
    feature_coefficients = feature_coefficients.reindex(feature_coefficients['Coefficient'].abs().sort_values(ascending=False).index)

    print("Feature Coefficients:")
    print(feature_coefficients)

def Logistic_stat(df):
    import statsmodels.api as sm
    import numpy as np, pandas as pd

    # Split data into features (X) and target variable (y)
    X_a = df.drop('converted', axis=1)  # Features
    y_a = df['converted']  # Target variable

    # Fit logistic regression model
    logit_model = sm.Logit(y_a, X_a)

    # Obtain the results of the logistic regression
    logit_result = logit_model.fit()

    # Print summary of the logistic regression results
    print(logit_result.summary())

    odds_ratios = np.exp(logit_result.params)
    conf_intervals = np.exp(logit_result.conf_int())

    # Create a DataFrame to display the odds ratios and their confidence intervals
    odds_ratios_df = pd.DataFrame({'Odds Ratio': odds_ratios, '95% CI Lower': conf_intervals[0], '95% CI Upper': conf_intervals[1]})
    print(odds_ratios_df)