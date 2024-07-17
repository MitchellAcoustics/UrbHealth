# Import the libraries needed for the analysis
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# obtain file names and append to an empty list
state_list = []
with os.scandir('statefiles') as statefolder:
    for entry in statefolder:
        if entry.name.endswith(".xlsx") and entry.is_file():
            state_list.append(entry.name)
state_list.sort()  # alphabetize list for clarity
print(state_list)

USA_df = pd.DataFrame()

# Regression and exportation for each state file
for states in state_list:
    # accounting that Florida has no DV data
    if states == 'Florida.xlsx':
        continue
    data = pd.read_excel(f'statefiles/{states}', engine='openpyxl')
    data = data.drop_duplicates(subset=['TractID'], keep=False)
    data.replace("NULL", pd.NA, inplace=True)
    # Selective imputation for int/float indices only
    fill = [c for c in data.columns if c not in ['State']]
    data = data.fillna(data[fill].mean())
    state_df = pd.DataFrame(data=data)
    pd.concat([USA_df, state_df])
    #print(USA_df.nunique)

    # Independent variables

    X = data[[
        'Stops per Sq Mile',
        'Park Area - Proportion',
        'est_ptrp',
        'est_vmiles',
        'lakids1share',
        'lahunv1share',
        'ht_ami',
        'emp_gravity',
        'Job Access By Transit']].copy()

    # VIF calculation - https://towardsdatascience.com/targeting-multicollinearity-with-python-3bd3b4088d0b
    X['intercept'] = 1  # add constant for statsmodels analysis

    # iteration cap = 25, VIF cap = 5
    # for i in range(25):
    #     vif = pd.DataFrame()
    #     vif["Variable"] = X.columns
    #     vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    #     vif = vif[vif['Variable'] != 'intercept']
    #
    #     # adjusting for NaN VIF values
    #     vif_na = vif[vif["VIF"].isna()]["Variable"].tolist()
    #     print()
    #     print("List of variables with scalar divide issue:")
    #     print(vif_na)
    #     X.drop(columns=vif_na, inplace=True)
    #     # print(X.head())
    #
    #     if vif["VIF"].max(numeric_only=True) < 5:
    #         print()
    #         print(f"Variance Inflation Factor Values for {states[:-5]}")
    #         print(vif)
    #         break
    #
    #     if vif["VIF"].max(numeric_only=True) > 5:
    #         vif_high = vif.loc[vif["VIF"].idxmax(), "Variable"]
    #         print()
    #         print(f"Iteration {i} of high VIF:")
    #         print(vif_high)
    #         X.drop(columns=[vif_high], inplace=True)

    # clean independent variables
    X.drop(columns=['intercept'], inplace=True)
    IVs = " + ".join([f"Q('{var}')" for var in X.columns])

    # dependent variables
    dependent = data[[
        'CHD',
        'Depression',
        'COPD',
        'DIABETES',
        'HIGHCHOL',
        'BPHIGH']]
    # Ensuring no IndexError
    data.reset_index(drop=True, inplace=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for Y in dependent.columns:
        y = data[Y].dropna()
        X_aligned = X.loc[y.index]  # Aligning X with y indices after dropping NaNs
        X_scaled_aligned = X_scaled[y.index]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled_aligned, y, test_size=0.3, random_state=42)

        # Initialize and fit the Random Forest Regressor
        rf = RandomForestRegressor(n_estimators=100)
        rf.fit(X_train, y_train)

        # Make predictions
        y_pred = rf.predict(X_test)

        # Print the regression performance metrics
        print(f"\nRandom Forest Regression Report for {Y} in {states[:-5]}:")
        print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
        print(f"R^2 Score: {r2_score(y_test, y_pred)}")

        # Feature importance - how relevant each factor is for prediction
        feature_importances = pd.DataFrame(rf.feature_importances_,
                                           index=X.columns,
                                           columns=['importance']).sort_values('importance', ascending=False)
        print(f"\nFeature importances for {Y} in {states[:-5]}:")
        print(feature_importances)

        result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
        perm_importance = pd.DataFrame(result.importances_mean, index=X.columns, columns=['importance']).sort_values(
            'importance', ascending=False)
        print("Permutation Importances:")
        print(perm_importance)
        fig, ax = plt.subplots()
        forest_importances = pd.Series(result.importances_mean, index=X.columns)
        forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
        ax.set_title(f"Feature importances using permutation on full model for {Y}")
        ax.set_ylabel("Mean accuracy decrease")
        fig.tight_layout()
        # plt.show()

data2 = pd.read_excel('merged_excel.xlsx')

X2 = data2[[
    'Stops per Sq Mile',
    'Park Area - Proportion',
    'est_ptrp',
    'est_vmiles',
    'lakids1share',
    'lahunv1share',
    'ht_ami',
    'emp_gravity',
    'Job Access By Transit']]

IVs = " + ".join([f"Q('{var}')" for var in X2.columns])

dependent = data2[[
    'CHD',
    'Depression',
    'COPD',
    'DIABETES',
    'HIGHCHOL',
    'BPHIGH']]

for Y2 in dependent.columns:
    dfY = data2[[Y2]]
    formula = f"{Y2} ~ 1 + {IVs}"
    # https://patsy.readthedocs.io/en/latest/quickstart.html
    fill = [c for c in data2.columns if c not in ['State']]
    data2 = data2.fillna(data2[fill].mean())
    mxd = smf.mixedlm(formula, data=data2, groups=data2["State"])
    regression = mxd.fit()
    print()
    print(f"Regression for {Y2} in US:")
    print(regression.summary())
formula = f"{Y2} ~ Q('Stops per Sq Mile') + Q('Park Area - Proportion') + Q('est_ptrp') + Q('est_vmiles')" \
    f" + Q('lakids1share') + Q('lahunv1share') + Q('ht_ami') + Q('emp_gravity')" \
    f" + Q('Job Access By Transit')"





