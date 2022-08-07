import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns


def set_fatty_liver_categories(stiffness):
    if stiffness < 7.5:
        return "Normal"
    elif 7.5 <= stiffness < 10:
        return "Moderate"
    elif 10 <= stiffness < 14:
        return "Severe"
    elif stiffness >= 14:
        return "Cirrhosis"



def adjust_income_values(income):
    if income == 1:
        return "0 to 4,999"
    elif income == 2:
        return "5,000 to 9,999"
    elif income == 3:
        return "10,000 to 14,999"
    elif income == 4:
        return "15,000 to 19,999"
    elif income == 5:
        return "20,000 to 24,999"
    elif income == 6:
        return "25,000 to 34,999"
    elif income == 7:
        return "35,000 to 44,999"
    elif income == 8:
        return "45,000 to 54,999"
    elif income == 9:
        return "55,000 to 64,999"
    elif income == 10:
        return "65,000 to 74,999"
    elif income == 12:
        return "20,000 and over"
    elif income == 13:
        return income == "Under 20,000"
    elif income == 14:
        return "75,000 to 99,999"
    elif income == 15:
        return "100,000 and Over"
    elif income == 77:
        return "Refused"
    elif income == 99:
        return "Don't Know"
    elif income == ".":
        return ""


def adjust_edu_values(edu):
    if edu == 1:
        return "Less than high school degree"
    elif edu == 2:
        return "High school / GED / AA degree"
    elif edu == 3:
        return "College graduate or above"
    elif edu == 7:
        return "Refused"
    elif edu == 9:
        return "Don't Know"
    elif edu == ".":
        return ""


def adjust_age_values(age):
    if age == 1:
        return "<20 years"
    elif age == 2:
        return "20-39 years"
    elif age == 3:
        return "40-59 years"
    elif age == 4:
        return "60+ years"
    elif age == ".":
        return ""

def adjust_gender_values(gender):
    if gender == 1:
        return "Male"
    elif gender == 2:
        return "Female"

if __name__ == '__main__':
    demographics = pd.read_csv('Z:/School/BIO-340/Project 1/DEMO_J.csv', encoding='latin1')
    liver_csv = pd.read_csv('Z:/School/BIO-340/Project 1/LUX_J.csv', encoding='latin1')
    dietary_csv =pd.read_csv('Z:/School/BIO-340/Project 1/DR1TOT_J.csv', encoding='latin1')
    demo_x = pd.DataFrame(demographics[['SEQN', 'DMDHRAGZ', 'DMDHREDZ', 'INDHHIN2', 'RIAGENDR']])
    diet_x = pd.DataFrame(dietary_csv[['SEQN', 'DR1TPROT', 'DR1_320Z']])
    liver_y = pd.DataFrame(liver_csv[['SEQN', 'LUXSMED']])

    # Merge the various csvs on SEQN
    data_df = demo_x.merge(liver_y, on='SEQN', how='inner')
    data_df = data_df.merge(diet_x, on='SEQN', how='inner')
    data_df.dropna(axis=0, inplace=True)

    # Standardize continuous variables
    scalar = StandardScaler()
    data_df['DR1TPROT'] = scalar.fit_transform(data_df[['DR1TPROT']])
    data_df['DR1_320Z'] = scalar.fit_transform(data_df[['DR1_320Z']])

    # Create dummy variables for the categorical variables.
    age = pd.get_dummies(data_df['DMDHRAGZ'].apply(adjust_age_values))
    education = pd.get_dummies(data_df['DMDHREDZ'].apply(adjust_edu_values))
    annual_income = pd.get_dummies(data_df['INDHHIN2'].apply(adjust_income_values), drop_first=True)
    gender = pd.get_dummies(data_df['RIAGENDR'].apply(adjust_gender_values))


    fatty_liver = data_df['LUXSMED'].apply(set_fatty_liver_categories)
    final_df = pd.concat([gender, education, annual_income, fatty_liver, data_df['DR1TPROT'], data_df['DR1_320Z']], axis=1)

    sns.countplot(x='LUXSMED', data=final_df).set(title='Countplot of fatty liver')
    plt.show()

    # Create outcome labels.
    labels = pd.DataFrame(final_df['LUXSMED'])
    labels.LUXSMED[labels.LUXSMED == 'Normal'] = 0
    labels.LUXSMED[labels.LUXSMED == 'Moderate'] = 1
    labels.LUXSMED[labels.LUXSMED == 'Severe'] = 1
    labels.LUXSMED[labels.LUXSMED == 'Cirrhosis'] = 1

    sns.countplot(x='LUXSMED', data=labels).set(title='Countplot of fatty liver')
    plt.show()

    labels = labels.apply(pd.to_numeric)
    # Drop non-useful columns and the outcome column
    final_df.drop(['LUXSMED' ,'Refused', 'Don\'t Know', '20,000 and over'], axis=1, inplace=True)
    final_df = final_df.apply(pd.to_numeric)

    # Logistic Regression
    x_train, x_test, y_train, y_test = train_test_split(
        final_df, labels, test_size=0.3, random_state=0
    )
    logmodel = LogisticRegression(multi_class='multinomial', class_weight='balanced')
    logmodel.fit(x_train, np.ravel(y_train))
    predict_log = logmodel.predict(x_test)
    print(classification_report(y_test, predict_log))

    importance = logmodel.coef_.flatten()
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.barh(final_df.columns, importance)
    plt.show()
# %%
