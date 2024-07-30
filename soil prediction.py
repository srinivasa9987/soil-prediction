
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import missingno as ms
import math
import scipy.stats as stats

data = pd.read_excel('/content/SOIL DATA GR (1).xlsx')
data.head().style.set_properties(**{'background-color':'blue','color':'white','border-color':'#8b8c8c'})

data.shape

data.isnull().sum()

data = data.drop("ID",axis='columns')

ms.bar(data, color = 'royalblue')
plt.show()

data.describe().style.background_gradient(cmap='tab20c')

data.describe().style.background_gradient(cmap='tab20c')



# Assuming data is a DataFrame with numerical columns
features = num_cols
n_bins = 50
histplot_hyperparams = {
    'kde': True,
    'alpha': 0.4,
    'stat': 'percent',
    'bins': n_bins
}

columns = features
n_cols = 4
n_rows = math.ceil(len(columns) / n_cols)
fig, ax = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
ax = ax.flatten()

handles = []
labels = []

for i, column in enumerate(columns):
    sns.histplot(data[column], ax=ax[i], color='#9E3F00', **histplot_hyperparams)
    ax[i].set_title(f'{column} Distribution')
    ax[i].set_xlabel(None)

    # Collect handles and labels for legend
    for handle, label in zip(*ax[i].get_legend_handles_labels()):
        if label not in labels:
            handles.append(handle)
            labels.append(label)
    ax[i].legend().remove()

# Turn off empty subplots
for i in range(i + 1, len(ax)):
    ax[i].axis('off')

fig.suptitle('Numerical Feature Distributions', ha='center', fontweight='bold', fontsize=25)
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.96), fontsize=25, ncol=3)
plt.tight_layout()
plt.show()

correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

X = data[['Sand %', 'Clay %', 'Silt %', 'EC mS/cm', 'O.M. %', 'CACO3 %',
       'N_NO3 ppm', 'P ppm', 'K ppm ', 'Mg ppm', 'Fe ppm', 'Zn ppm', 'Mn ppm',
       'Cu ppm', 'B ppm']]
y = data['pH']


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Model Interpretation
# Print model coefficients
print('Model Coefficients:')
for feature, coef in zip(X.columns, model.coef_):
    print(f'{feature}: {coef:.4f}')

from sklearn.impute import SimpleImputer

# Create an imputer
imputer = SimpleImputer(strategy='mean')

# Fit and transform the imputer on X_test
X_test_imputed = imputer.fit_transform(X_test)

# Now you can make predictions with the imputed data
y_pred = model.predict(X_test_imputed)

X_test_cleaned = X_test.dropna()
y_test_cleaned = y_test[X_test_cleaned.index]


y_pred = model.predict(X_test_cleaned)
predicted_classes = [classify_ph(val) for val in y_pred]


mse = mean_squared_error(y_test_cleaned, y_pred)
r2 = r2_score(y_test_cleaned, y_pred)
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'R-squared (R2): {r2:.4f}')

def classify_ph(value):
    if value < 5.3:
        return 'Acute Alkaline soil'
    elif 6.3 <= value <= 7.3:
        return 'Good Soil'
    elif value > 7.3:
        return 'acute acidic soil'
    else:
        return 'normal soil'

predicted_classes = [classify_ph(val) for val in y_pred]



# Function to predict pH value based on user input features
def predict_ph():
    try:
        sand = float(input("Enter Sand %: "))
        clay = float(input("Enter Clay %: "))
        silt = float(input("Enter Silt %: "))
        ec = float(input("Enter EC mS/cm: "))
        om = float(input("Enter O.M. %: "))
        caco3 = float(input("Enter CACO3 %: "))
        n_no3 = float(input("Enter N_NO3 ppm: "))
        p_ppm = float(input("Enter P ppm: "))
        k_ppm = float(input("Enter K ppm: "))
        mg_ppm = float(input("Enter Mg ppm: "))
        fe_ppm = float(input("Enter Fe ppm: "))
        zn_ppm = float(input("Enter Zn ppm: "))
        mn_ppm = float(input("Enter Mn ppm: "))
        cu_ppm = float(input("Enter Cu ppm: "))
        b_ppm = float(input("Enter B ppm: "))

        user_data = pd.DataFrame({
            'Sand %': [sand], 'Clay %': [clay], 'Silt %': [silt], 'EC mS/cm': [ec], 'O.M. %': [om],
            'CACO3 %': [caco3], 'N_NO3 ppm': [n_no3], 'P ppm': [p_ppm], 'K ppm ': [k_ppm], 'Mg ppm': [mg_ppm],
            'Fe ppm': [fe_ppm], 'Zn ppm': [zn_ppm], 'Mn ppm': [mn_ppm], 'Cu ppm': [cu_ppm], 'B ppm': [b_ppm]
        })

        predicted_ph = model.predict(user_data)[0]
        classification = classify_ph(predicted_ph)

        print(classification)
    except ValueError:
        print("Invalid input. Please enter numerical values.")

# Predict pH value based on user input
predict_ph()

