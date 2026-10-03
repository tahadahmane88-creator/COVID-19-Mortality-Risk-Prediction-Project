import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import joblib

df = pd.read_csv('simple_clean_data.csv')
age_moralyty = df.groupby('AGE')['DIED'].mean() * 100
plt.figure(figsize=(8, 4))
plt.plot(age_moralyty.index , age_moralyty.values, color= 'Red')
plt.title("mortaloty by age")
plt.xlabel('age')
plt.ylabel('mortalyti en %')
plt.grid(True)
plt.show()

diabities_df = df[df['DIABETES'].isin([1,2])]
diabities_mortalyti = df.groupby('DIABETES')['DIED'].mean()*100
plt.figure(figsize=(6, 4))
plt.bar(['Diabétique ', 'Non-diabétique '], diabities_mortalyti.values, color=['orange', 'green'])
plt.title("taux de mort pour les dieabets")
plt.ylabel("taux de mort en %")
plt.show()

conditions = ['PNEUMONIA', 'DIABETES', 'HIPERTENSION', 'OBESITY']

for condition in conditions:
    subset = df[df[condition].isin([1, 2])]

    rates = subset.groupby(condition)['DIED'].mean() * 100

    print(f"=== {condition} ===")
    print(f"malad (1)     : {rates.get(1, 0):.1f}%")
    print(f"non malad (2) : {rates.get(2, 0):.1f}%\n")

x = df[['AGE', 'PNEUMONIA', 'DIABETES', 'HIPERTENSION', 'OBESITY', 'PATIENT_TYPE']]
y = df['DIED']
x_train , x_test, y_train, y_test = train_test_split(
    x, y , train_size=0.2, random_state=42 , stratify=y
)
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=20)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test , y_pred)
print(f"the accuracy is {acc*100 :.2f}%")
new_patient = pd.DataFrame([{
    'AGE': 65,
    'PNEUMONIA': 1,
    'DIABETES': 2,
    'HIPERTENSION': 1,
    'OBESITY': 2,
    'PATIENT_TYPE': 2
}])
proba_danger = model.predict_proba(new_patient)[0][1] * 100

print(f" pourcentage de mourire : {proba_danger:.1f}%")


joblib.dump(model, 'covid_risk_model.pkl')
print("le model est enregsitrer dans 'covid_risk_model.pkl'")




