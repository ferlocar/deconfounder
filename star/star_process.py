# %%
import pandas as pd
import numpy as np
import json

star_data = pd.read_spss("../data/STAR_Students.sav")

columns = ['stdntid', 'gender', 'race', 'birthmonth', 'birthday', 'birthyear', \
           'gkfreelunch', 'gktreadss', 'gktmathss', 'gktlistss']
for c in star_data.columns:
    if c.startswith('g1'):
        columns.append(c)

df = star_data[columns]

# %%
star_covars = json.load(open("star_variables.json"))
cat_covar_columns, categories = list(zip(*star_covars.items()))
cat_covar_columns, categories = list(cat_covar_columns), list(categories)

# %%
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(
    categories=categories, 
    handle_unknown='use_encoded_value',
    unknown_value=np.nan
)

remain_columns = list(set(columns) - set(cat_covar_columns))
encoded_df = encoder.fit_transform(df[cat_covar_columns])+1
encoded_df = pd.DataFrame(encoded_df, columns=cat_covar_columns)

for c in encoded_df.columns:
    if (pd.isna(encoded_df[c]).sum()) != (pd.isna(df[c]).sum()):
        print(c)

encoded_df = pd.concat([encoded_df, df[remain_columns]], axis=1)
encoded_df = encoded_df[columns]

# %%
encoded_df.to_csv("../data/STAR_Students_Encoded.csv", index=False)

# %%
