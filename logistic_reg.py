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

def ROC_AUC(df):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import roc_curve, auc

    # Split data into predictor variables (X) and output variable (y)
    X = df.drop('converted', axis=1)  
    y = df['converted']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit logistic regression model
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    # Predict probabilities for the test set
    y_prob = logreg.predict_proba(X_test)[:, 1]

    # Compute false positive rate (FPR) and true positive rate (TPR) for ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    # Compute Area Under the Curve (AUC) for ROC curve
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

    # Check coefficients for linearity
    print("Coefficients of predictor variables:")
    print(logreg.coef_)

def shapiro_linearity(df):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures
    from scipy.stats import shapiro

    # Split data into predictor variables (X) and output variable (y)
    X = df.drop('converted', axis=1)  # Assuming 'converted' is the output variable
    y = df['converted']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit logistic regression model
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    # Predict probabilities for the test set
    y_prob = logreg.predict_proba(X_test)[:, 1]

    # Compute residuals
    residuals = y_test - y_prob

    # Plot residuals
    plt.figure(figsize=(8, 6))
    plt.scatter(logreg.predict(X_test), residuals, c='b', alpha=0.5)
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.show()

    # Check for normality of residuals
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=20, density=True, color='blue', alpha=0.7)
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title('Histogram of Residuals')
    plt.show()

    # Perform Shapiro-Wilk test for normality
    shapiro_stat, shapiro_pval = shapiro(residuals)
    print("Shapiro-Wilk test for normality:")
    print("Test statistic:", shapiro_stat)
    print("p-value:", shapiro_pval)

def Chisquare(data,column='converted',column2='',P_value=0.05):

    import scipy.stats as stats
    import numpy as np, pandas as pd
    from scipy.stats import chi2_contingency

    group1=data[str(column2)][data[str(column)]>0].reset_index(drop=True)
    group2=data[str(column2)][data[str(column)]==0].reset_index(drop=True)
    observed_data=group1.value_counts().sort_index()
    expected_data=group2.value_counts().sort_index()
    o_d='['
    for o in observed_data:o_d=o_d+str(o)+', '
    o_d=o_d+']'
    o_d=o_d.replace(', ]',']')
    e_d='['
    for e in expected_data:e_d=e_d+str(e)+', '
    e_d=e_d+']'
    e_d=e_d.replace(', ]',']')

    datatot_e=[]
    datatot_o=[]
    for e in expected_data: datatot_e.append(e)
    for o in observed_data: datatot_o.append(o)
    datatot=[datatot_o,datatot_e]

    stat, p, dof, expected = chi2_contingency(datatot)

    print('\nWe are now checking whether there is relationhip between the values in the column "'+str(column2)+'" when "'+str(column)+'" is equal to "1" : '+str(o_d)+' and the values in the same column when "'+str(column)+'" is equal to "0" : '+str(e_d)+'.\n')
    print('\nIn this case the  p-value is: '+str(round(float(p),3))+'.\n', 'chi_square_test_statistic is : ' + str(round(stat,3))+'.\n', 'degree of freedom are: '+str(dof))
    if round(float(p),3) < P_value: message='\nThese results indicate that there is statistically significant relationship between the variable "'+str(column)+'" and "'+str(column2)+'".'
    else: message='These results indicate that there is NOT statistically significant relationship between the variable "'+str(column)+'" and "'+str(column2)+'".'
    print(message)

def IndSamplesTtest(data,column1,column2,P_value,tails):
    import pandas as pd , numpy as np, scipy.stats as stats
    from statsmodels.stats.weightstats import ttest_ind

    group1=data[str(column1)][data[str(column2)] > 0].reset_index(drop=True)
    group2=data[str(column1)][data[str(column2)] == 0].reset_index(drop=True)
    
    #Check variance
    if np.var(group1)>np.var(group2):
        if (round(np.var(group1)/np.var(group2))) < 4: eq_var=True
        else: eq_var=False
    else:
        if (round(np.var(group2)/np.var(group1))) < 4: eq_var=True
        else: eq_var=False

    t_statistic, p_value = stats.ttest_ind(a=group1, b=group2, equal_var=eq_var)
    if tails==1: p_value=p_value/2
    print('\nAn independent samples t-test is used when you want to compare the means of a normally distributed interval dependent variable for two independent groups.\n'+'\nWe are now checking whether the average of column "'+str(column1)+'" differs significantly between the gorups in column "'+str(column2)+'"\n', '\nIn this case the t-statistic score is: '+str(round(t_statistic,3))+' and the p-value is: '+str(round(float(p_value),4)))
    if round(float(p_value),4) < P_value:
        message='\nThe mean of the variable "'+str(column1)+'" for group 1 (i.e. when "'+str(column2)+'" is equal to 1) is '+str(round(group1.mean(),2))+', which is statistically significantly different from the mean of group 2 (i.e. when "'+str(column2)+'" is equal to 0): '+str(round(group2.mean(),2))+' for a P-Value<'+str(P_value)+'.\n'
    else:
        message='\nThe mean of the variable "'+str(column1)+'" for group 1 (i.e. when "'+str(column2)+'" is equal to 1) is '+str(round(group1.mean(),2))+', which IS NOT statistically significantly different from the mean of group 2 (i.e. when "'+str(column2)+'" is equal to 0): '+str(round(group2.mean(),2))+' for a P-Value<'+str(P_value)+'.\n'
    print(message)

def mannwhitneyutest(data,column1,column2,P_value,tails):
    import pandas as pd , numpy as np, scipy.stats as stats
    from scipy.stats import mannwhitneyu

    group1=data[str(column1)][data[str(column2)] > 0].reset_index(drop=True)
    group2=data[str(column1)][data[str(column2)] == 0].reset_index(drop=True)
    
    #Check variance
    if np.var(group1)>np.var(group2):
        if (round(np.var(group1)/np.var(group2))) < 4: eq_var=True
        else: eq_var=False
    else:
        if (round(np.var(group2)/np.var(group1))) < 4: eq_var=True
        else: eq_var=False

    t_statistic, p_value = mannwhitneyu(group1, group2)
    if tails==1: p_value=p_value/2
    print('\nThe Wilcoxon-Mann-Whitney test is a non-parametric analog to the independent samples t-test and can be used when you do not assume that the dependent variable is a normally distributed interval variable (you only assume that the variable is at least ordinal).\n'+'\nWe are now checking whether the average of column "'+str(column1)+'" differs significantly between the gorups in column "'+str(column2)+'"\n', '\nIn this case the t-statistic score is: '+str(round(t_statistic,3))+' and the p-value is: '+str(round(float(p_value),4)))
    if round(float(p_value),4) < P_value:
        message='\nThe mean of the variable "'+str(column1)+'" for group 1 (i.e. when "'+str(column2)+'" is equal to 1) is '+str(round(group1.mean(),2))+', which is statistically significantly different from the mean of group 2 (i.e. when "'+str(column2)+'" is equal to 0): '+str(round(group2.mean(),2))+' for a P-Value<'+str(P_value)+'.\n'
    else:
        message='\nThe mean of the variable "'+str(column1)+'" for group 1 (i.e. when "'+str(column2)+'" is equal to 1) is '+str(round(group1.mean(),2))+', which IS NOT statistically significantly different from the mean of group 2 (i.e. when "'+str(column2)+'" is equal to 0): '+str(round(group2.mean(),2))+' for a P-Value<'+str(P_value)+'.\n'
    print(message)