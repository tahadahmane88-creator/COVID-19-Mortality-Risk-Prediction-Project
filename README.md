# 🦠 COVID-19 Mortality Risk Prediction System

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

Ce projet met en œuvre une solution de **Machine Learning** permettant d'estimer le risque de mortalité des patients atteints du COVID-19 en fonction de leur âge, de leurs comorbidités (diabète, hypertension, pneumonie, obésité) et de leur mode de prise en charge hospitalière.

---

## 📌 Aperçu du Projet

L'objectif est de fournir un outil prédictif d'aide à la décision médicale afin d'identifier rapidement les patients à haut risque nécessitant une prise en charge prioritaire.

- **Jeu de données** : Données épidémiologiques (~1 048 575 enregistrements).
- **Modèle IA** : *Random Forest Classifier*.
- **Performances** : **93.26% d'Accuracy** et un score **ROC-AUC de 0.959**.

---

## 📊 Principales Découvertes (Exploratory Data Analysis)

L'analyse des données a révélé des facteurs déterminants dans la sévérité des cas :
1. **Hospitalisation (`PATIENT_TYPE`)** : C'est la variable la plus prédictive du niveau de gravité du patient.
2. **Pneumonie (`PNEUMONIA`)** : La présence d'une pneumonie augmente le taux de mortalité de **2.5%** à **38.5%**.
3. **Âge (`AGE`)** : Hausse exponentielle du risque de mortalité au-delà de 50 à 60 ans.
4. **Comorbidités (`DIABETES`, `HIPERTENSION`)** : Le diabète multiplie le risque par plus de 4 (**22.6%** chez les diabétiques vs **5.3%** chez les non-diabétiques).

---

## 📁 Structure du Dépôt

```text
├── .gitignore               # Fichiers ignorés (données .csv lourdes et configurations IDE)
├── README.md                # Documentation officielle du projet
├── requirements.txt         # Dépendances Python nécessaires
├── script.py                # Pipeline complet (Nettoyage, EDA, Entraînement, Évaluation)
└── covid_risk_model.pkl     # Modèle Random Forest entraîné et sauvegardé
