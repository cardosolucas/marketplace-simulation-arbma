import pandas as pd

dataset = pd.read_csv('datasets_arbma/compas-scores-raw.csv', header=0)

dataset = dataset[dataset['DisplayText'] == 'Risk of Violence']

dataset = dataset[['Sex_Code_Text', 'Ethnic_Code_Text', 'LegalStatus', 'CustodyStatus', 'MaritalStatus', 'DecileScore', 'Screening_Date', 'DateOfBirth']]

dataset['Screening_Date'] = pd.to_datetime(dataset['Screening_Date'])
dataset['DateOfBirth'] = pd.to_datetime(dataset['DateOfBirth'])

dataset['Age_At_Screening'] = (
    dataset['Screening_Date'].dt.year 
    - dataset['DateOfBirth'].dt.year 
    - (
        (dataset['Screening_Date'].dt.month < dataset['DateOfBirth'].dt.month) | 
        ((dataset['Screening_Date'].dt.month == dataset['DateOfBirth'].dt.month) & 
         (dataset['Screening_Date'].dt.day < dataset['DateOfBirth'].dt.day))
      ).astype(int)
)

bins = [0, 24, 45, 200]

labels = ['Less than 25', '25 - 45', 'Greater than 45']

dataset['age'] = pd.cut(dataset['Age_At_Screening'], bins=bins, labels=labels)

dataset.drop(['Screening_Date', 'DateOfBirth', 'Age_At_Screening'], inplace=True, axis=1)

dataset = dataset[dataset['Ethnic_Code_Text'].isin(['Caucasian', 'African-American'])]

dataset['Sex_Code_Text'] = dataset['Sex_Code_Text'].map({'Male': 0, 'Female': 1})
dataset['Ethnic_Code_Text'] = dataset['Ethnic_Code_Text'].map({'Caucasian': 0, 'African-American': 1})

columns_to_encode = ['LegalStatus', 'CustodyStatus', 'MaritalStatus', 'age']

dataset = pd.get_dummies(dataset, columns=columns_to_encode, dtype=int)
