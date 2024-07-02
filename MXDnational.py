# Import the libraries needed for the analysis
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf
import os

# obtain file names and append to an empty list
state_list = []
with os.scandir('/Users/dylanmach/Downloads/statefiles') as statefolder:
    for entry in statefolder:
        if entry.name.endswith(".xlsx") and entry.is_file():
            state_list.append(entry.name)
state_list.sort()  # alphabetize list for clarity
print(state_list)

data = pd.read_excel('/Users/dylanmach/Downloads/merged_excel.xlsx')
data = data.drop_duplicates(subset=['TractID'], keep=False)
data.replace("NULL", pd.NA, inplace=True)
# Selective imputation for int/float indices only
fill = [c for c in data.columns if c not in ['State']]
data = data.fillna(data[fill].mean())
state_df = pd.DataFrame(data=data)

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

dependent = data[[
    'CHD',
    'Depression',
    'COPD',
    'DIABETES',
    'HIGHCHOL',
    'BPHIGH']]

# VIF calculation - https://towardsdatascience.com/targeting-multicollinearity-with-python-3bd3b4088d0b
X['intercept'] = 1  # add constant for statsmodels analysis

# iteration cap = 25, VIF cap = 5
for i in range(25):
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif[vif['Variable'] != 'intercept']

    # adjusting for NaN VIF values
    vif_na = vif[vif["VIF"].isna()]["Variable"].tolist()
    print()
    print("List of variables with scalar divide issue:")
    print(vif_na)
    X.drop(columns=vif_na, inplace=True)
    # print(X.head())

    if vif["VIF"].max(numeric_only=True) < 5:
        print()
        print(f"Variance Inflation Factor Values for USA")
        print(vif)
        break

    if vif["VIF"].max(numeric_only=True) > 5:
        vif_high = vif.loc[vif["VIF"].idxmax(), "Variable"]
        print()
        print(f"Iteration {i} of high VIF:")
        print(vif_high)
        X.drop(columns=[vif_high], inplace=True)

X.drop(columns=['intercept'], inplace=True)
IVs = " + ".join([f"Q('{var}')" for var in X.columns])

for Y in dependent.columns:
    dfY = data[[Y]]
    formula = f"{Y} ~ 1 + {IVs}"
    # https://patsy.readthedocs.io/en/latest/quickstart.html
    mxd = smf.mixedlm(formula, data=data, groups=data["State"])
    regression = mxd.fit()
    print()
    print(f"Regression for {Y} in US:")
    print(regression.summary())
formula = f"{Y} ~ Q('Stops per Sq Mile') + Q('Park Area - Proportion') + Q('est_ptrp') + Q('est_vmiles')" \
    f" + Q('lakids1share') + Q('lahunv1share') + Q('ht_ami') + Q('emp_gravity')" \
    f" + Q('Job Access By Transit')"

# Regression and exportation for each state file
# for states in state_list:
#     # accounting that Florida has no DV data
#     if states == 'Florida.xlsx':
#         continue
#     data = pd.read_excel(f'/Users/dylanmach/Downloads/statefiles/{states}', engine='openpyxl')
#     data = data.drop_duplicates(subset=['TractID'], keep=False)
#     data.replace("NULL", pd.NA, inplace=True)
#     # Selective imputation for int/float indices only
#     fill = [c for c in data.columns if c not in ['State']]
#     data = data.fillna(data[fill].mean())
#     state_df = pd.DataFrame(data=data)
#     pd.concat([USA_df, state_df])
#     print(USA_df.nunique)
#
#     X = data[[
#         'Stops per Sq Mile',
#         'Park Area - Proportion',
#         'est_ptrp',
#         'est_vmiles',
#         'lakids1share',
#         'lahunv1share',
#         'ht_ami',
#         'emp_gravity',
#         'Job Access By Transit']].copy()
#
#     # VIF calculation - https://towardsdatascience.com/targeting-multicollinearity-with-python-3bd3b4088d0b
#     X['intercept'] = 1  # add constant for statsmodels analysis
#
#     # iteration cap = 25, VIF cap = 5
#     # for i in range(25):
#     #     vif = pd.DataFrame()
#     #     vif["Variable"] = X.columns
#     #     vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
#     #     vif = vif[vif['Variable'] != 'intercept']
#     #
#     #     # adjusting for NaN VIF values
#     #     vif_na = vif[vif["VIF"].isna()]["Variable"].tolist()
#     #     print()
#     #     print("List of variables with scalar divide issue:")
#     #     print(vif_na)
#     #     X.drop(columns=vif_na, inplace=True)
#     #     # print(X.head())
#     #
#     #     if vif["VIF"].max(numeric_only=True) < 5:
#     #         print()
#     #         print(f"Variance Inflation Factor Values for {states[:-5]}")
#     #         print(vif)
#     #         break
#     #
#     #     if vif["VIF"].max(numeric_only=True) > 5:
#     #         vif_high = vif.loc[vif["VIF"].idxmax(), "Variable"]
#     #         print()
#     #         print(f"Iteration {i} of high VIF:")
#     #         print(vif_high)
#     #         X.drop(columns=[vif_high], inplace=True)
#
#     # clean independent variables
#     X.drop(columns=['intercept'], inplace=True)
#     IVs = " + ".join([f"Q('{var}')" for var in X.columns])
#
#     # dependent variables
#     dependent = data[[
#         'CHD',
#         'Depression',
#         'COPD',
#         'DIABETES',
#         'HIGHCHOL',
#         'BPHIGH']]
#     # Ensuring no IndexError
#     data.reset_index(drop=True, inplace=True)
