# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:43:53 2023

@author: kaouther.mimouni_ver
"""
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
df = pd.read_csv("kaggle_survey_2020_responses.csv", sep =",", low_memory = False)
#Suppression de la ligne 0 (inutile) sur df et data (utilisé pour dataviz)
questions=df.iloc[0,:]

df = df.drop(df.index[0])
data=df
dataemmanuel=df
df_prep = pd.read_csv("df_preprocessed_new.csv")
#On ne garde que 5 métiers en valeur cible
df_datajob = df[(df.Q5 == 'Data Analyst')| (df.Q5 == 'Data Engineer') | (df.Q5 == 'Data Scientist') | (df.Q5 == 'DBA/Database Engineer') |(df.Q5 == 'Machine Learning Engineer')]
df_datajob.head()

st.sidebar.title("Sommaire")
pages=["Contexte du projet","Exploration des données","Analyse de données globale","Zoom sur les 3 métiers","Jeu des données post nettoyage", "Modélisation","Conclusion & Perspectives"]
page=st.sidebar.radio("Aller vers la page : ", pages)
if page == pages[0] :
    st.image("datajobdatasc.png")
    st.write("### Contexte du projet")
    st.write("Ce projet s'inscrit dans le contexte de la formation de Data Analyst. L’objectif est de comprendre à l’aide des données les différents profils techniques qui se sont créés dans l’industrie de la Data. Plus exactement, il faudra mener une analyse poussée des tâches effectuées ainsi que des outils utilisés par chaque poste afin d’établir des ensembles de compétences et outils correspondant à chaque poste du monde de la Data.")
    st.write("La base de données provient d'un sondage réalisé en 2020 pour comprendre l'état de la data science et du machine learning à cette époque. Les résultats incluent des chiffres bruts concernant les personnes qui travaillent avec les données, comment est utilisé le machine learning dans différents secteurs d'activité, et les meilleures compétences pour s'intégrer dans le domaine.")
    st.write("Ce sondage, mené sur une durée de 3,5 semaines en octobre 2020, a récolté 20 036 réponses après nettoyage des données. Bien que des sondages similaires aient été réalisés en 2017, 2018, 2019, 2021 et 2022, nous nous sommes concentrés uniquement sur les données de 2020.")
    st.write("Dans un premier temps, nous explorerons ce dataset. Puis nous l'analyserons visuellement pour en extraire des informations selon certains axes d'étude. Finalement nous implémenterons des modèles de Machine Learning qui pourrait permettre de construire un système de recommandation de poste permettant aux autres apprenants de viser exactement le poste correspondant le plus à leurs appétences.")
    #st.image("datajob.jpg")
    
elif page == pages[1]:

    st.write("### Exploration des données")
    st.write("#### Principales caractéristiques du dataset :")
    st.write("Un très grand nombre de variables:")
    st.write("- Presque toutes les variables sont des variables catégorielles")
    st.write("- Un très grand pourcentage de NAN sur certaines variables, mais qui provient du fait que les réponses des questions à choix multiples ont des questions à choix multiples ont été transformée en plusieurs variables binaires selon le principe du one-hot encoding")
    st.write("- Les noms des variables sont codés et difficilement interprétables : Q1 What is your age (# years)?")
    st.write("- Certaines variables sont liées les unes aux autres")
    st.write("#### Aperçu du Dataframe:")
    #st.image("Template_Excel.png")
    st.dataframe(df.head())
    st.write("Dimensions du dataframe :")
    st.write(df.shape)
    if st.checkbox("Afficher les valeurs manquantes") :
        # Calculer le taux de valeurs manquantes
        missing_ratio = df.isna().mean()

        st.dataframe(missing_ratio)
    if st.checkbox("Afficher les doublons") :
        st.write(df.duplicated().sum())
    if st.checkbox("Aperçu des questions") :
        st.dataframe(questions) 
        
elif page == pages[2]:
    st.header("Analyse de données globale")
    #Segmentation valeur cible
    seg1=[['Data Analyst','Data Analyst'],    ['Business Analyst','Other'],       ['Data Engineer','Data Engineer'],        ['DBA/Database Engineer','DBA/Database Engineer'],
           ['Software Engineer','Other'],           ['Statistician','Other'],   ['Product/Project Manager','Other'],         ['Research Scientist','Other'],
           ['Student','Other'],  ['Data Scientist','Data Scientist'],            ['Machine Learning Engineer','ML Engineer'],           ['Other','Other'],            ['','Other'],
           ['Currently not employed','Other']]
    cnty1 = pd.DataFrame(seg1, columns =['Q5', 'Segm'])
    survey=data.merge(cnty1, on='Q5', how='left')
    survey['Segm'].fillna("Other", inplace = True)
    data_q5 = survey[[i for i in survey.columns if 'Segm' in i]]
    data_q5_count = pd.Series(dtype='int')

    for i in data_q5.columns:

        data_q5_count[data_q5[i].value_counts().index[0]] = data_q5[i].count()

    data_q5_count=data_q5.Segm.value_counts()
    fig01, ax = plt.subplots(1,1, figsize=(10, 10))

    #st.write("### Analyse de données")
    
    #Segmentation valeur cible
    seg1=[['Data Analyst','Data Analyst'],    ['Business Analyst','Other'],       ['Data Engineer','Data Engineer'],        ['DBA/Database Engineer','DBA/Database Engineer'],
           ['Software Engineer','Other'],           ['Statistician','Other'],   ['Product/Project Manager','Other'],         ['Research Scientist','Other'],
           ['Student','Other'],  ['Data Scientist','Data Scientist'],            ['Machine Learning Engineer','ML Engineer'],           ['Other','Other'],            ['','Other'],
           ['Currently not employed','Other']]
    cnty1 = pd.DataFrame(seg1, columns =['Q5', 'Segm'])
    survey=data.merge(cnty1, on='Q5', how='left')
    survey['Segm'].fillna("Other", inplace = True)
    data_q5 = survey[[i for i in survey.columns if 'Segm' in i]]
    data_q5_count = pd.Series(dtype='int')

    for i in data_q5.columns:

        data_q5_count[data_q5[i].value_counts().index[0]] = data_q5[i].count()

    data_q5_count=data_q5.Segm.value_counts()
    plt.pie(data_q5_count, labels=data_q5_count.index,colors=sns.color_palette("crest", 7))

    fig4, ax = plt.subplots(1,1, figsize=(20, 10))
    total=data_q5_count.sum()
    percent = data_q5_count/total*100

    new_labels = [i+'  {:.2f}%'.format(j) for i, j in zip(data_q5_count.index, percent)]

    plt.barh(data_q5_count.index, data_q5_count, color=sns.color_palette("crest", 7),  edgecolor='darkgray')
    plt.yticks(range(len(data_q5_count.index)), new_labels, va="center",fontsize=20)
    plt.tight_layout()

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.axes.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", left=False)
    #ax.grid(axis='y', linestyle='-', alpha=0.4)
    st.write("##### Variable cible: Title")
    st.pyplot(fig4)
    #fig40, ax = plt.subplots(1,1, figsize=(10, 10))
    #ax.pie(data_q5_count, labels=data_q5_count.index,autopct='%1.1f%%', normalize=True,shadow=True,colors=sns.color_palette("crest", 7))
    #st.pyplot(fig40)
    # 'Education Distribution'
    q4_order = [
      'No formal education past high school',
      'Professional degree',
      'Some college/university study without earning a bachelor’s degree',
      'Bachelor’s degree',
      'Master’s degree',
      'Doctoral degree',
      'I prefer not to answer'
    ]

    data_q4 = data['Q4'].value_counts()[q4_order]

    fig, ax = plt.subplots(1,1, figsize=(12, 6))
    # Hide grid lines
    ax.grid(False)
    total=19569
    percent = data_q4/total*100

    new_labels = [i+'  {:.2f}%'.format(j) for i, j in zip(data_q4.index, percent)]

    plt.barh(data_q4.index, data_q4, color=sns.color_palette("crest", 7),  edgecolor='darkgray')
    plt.yticks(range(len(data_q4.index)), new_labels,fontsize=14)
    plt.tight_layout()

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.axes.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", left=False)
    #fig.text(0.13, 0.95, 'Education Distribution', fontsize=15, fontweight='bold', fontfamily='serif')
    st.write("##### Niveau d'éducation")
    st.pyplot(fig)
    
    # 'Compensation'

    compensation_level = {"$0-999": "< 1000$", "1,000-1,999": "1000$ - 5000$","2,000-2,999": "1000$ - 5000$",
        "3,000-3,999": "1000$ - 5000$","4,000-4,999": "1000$ - 5000$","5,000-7,499": "5000$ - 10000$",
        "7,500-9,999": "5000$ - 10000$","10,000-14,999": "10000$ - 50000$",
        "15,000-19,999": "10000$ - 50000$","20,000-24,999": "10000$ - 50000$",
        "25,000-29,999": "10000$ - 50000$","30,000-39,999": "10000$ - 50000$",
        "40,000-49,999": "10000$ - 50000$","50,000-59,999": "50000$ - 100000$",
        "60,000-69,999": "50000$ - 100000$","70,000-79,999":"50000$ - 100000$",
        "80,000-89,999": "50000$ - 100000$","90,000-99,99": "50000$ - 100000$",
        "100,000-124,999": "100000$ - 500000$","125,000-149,999": "100000$ - 500000$",
        "150,000-199,999": "100000$ - 500000$","200,000-249,999": "100000$ - 500000$",
        "250,000-299,999": "100000$ - 500000$","300,000-500,000": "100000$ - 500000$",
        "> $500,000": "> 500000$",}

    data['Q24'] = data['Q24'].replace(compensation_level )
    q24_order = ['< 1000$','1000$ - 5000$', '5000$ - 10000$', '10000$ - 50000$', '50000$ - 100000$',
                 '100000$ - 500000$', '> 500000$']

    data_q24 = data['Q24'].value_counts()[q24_order]
    fig6, ax = plt.subplots(1,1, figsize=(12, 6))
    total=data_q24.sum()
    percent = data_q24/total*100

    new_labels = [i+'  {:.2f}%'.format(j) for i, j in zip(data_q24.index, percent)]

    plt.barh(data_q24.index, data_q24, color=sns.color_palette("crest", 7),  edgecolor='darkgray')
    plt.yticks(range(len(data_q24.index)), new_labels, va="center",fontsize=14)
    plt.tight_layout()

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.axes.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", left=False)
    st.write("##### Rémunération actuelle")
    st.pyplot(fig6)

    #'ML Experience Distribution'
    data_datajob = data[(data.Q5 == 'Data Analyst')| (data.Q5 == 'Data Engineer') | (data.Q5 == 'Data Scientist') | (data.Q5 == 'DBA/Database Engineer') |(data.Q5 == 'Machine Learning Engineer')]
    data_datajob.head()

    data_datajob.Q15 = data_datajob.Q15.replace("I do not use machine learning methods", "none")
    data_datajob.Q15.unique()
    order = ["none","Under 1 year","1-2 years", "2-3 years", "3-4 years", "4-5 years","5-10 years","10-20 years", "20 or more years"]
    fig11, ax = plt.subplots(1,1, figsize=(15, 6))
    data_q15 = data_datajob['Q15'].value_counts()[order]
    total=data_q15.sum()
    percent = data_q15/total*100
    new_labels = [i+'  {:.2f}%'.format(j) for i, j in zip(order, percent)]

    plt.barh(order, data_q15, color=sns.color_palette("crest", 7),  edgecolor='darkgray')
    plt.yticks(range(len(order)), new_labels, va="center",fontsize=14)
    plt.tight_layout()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", left=False)
    st.write("##### Années d'utilisation des méthodes de Marchine learning")
    st.pyplot(fig11)

    #BI tool le plus utilisé , total et par métier
    data_q32=data['Q32'].value_counts()
    fig9 = plt.figure(figsize=(10,10))
    total=data_q32.sum()
    percent = data_q32/total*100
    new_labels = [i+'  {:.2f}%'.format(j) for i, j in zip(data_q32.index, percent)]

    plt.barh(data_q32.index, data_q32, color=sns.color_palette("crest", 7),  edgecolor='darkgray')
    plt.yticks(range(len(data_q32.index)), new_labels, va="center",fontsize=14)
    plt.tight_layout()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    #ax.tick_params(axis="y", left=False)
    #plt.title('BI tool le plus utilisé , total et par métier',fontsize=15, fontweight='bold', fontfamily='serif');
    st.write("##### Outils BA le plus utilisé")
    st.pyplot(fig9)
  

    #Langage recommandé en priorité
    fig13, ax = plt.subplots(1,1, figsize=(20, 20))
    data_q8=data['Q8'].value_counts()
    #fig99 = plt.figure(figsize=(10,10))
    total=data_q8.sum()
    percent = data_q8/total*100
    new_labels = [i+'  {:.2f}%'.format(j) for i, j in zip(data_q8.index, percent)]

    plt.barh(data_q8.index, data_q8, color=sns.color_palette("crest", 7),  edgecolor='darkgray')
    plt.yticks(range(len(data_q8.index)), new_labels, va="center",fontsize=20)
    plt.tight_layout()
    for spine in ax.spines.values():
        spine.set_visible(False)
    st.write("##### Langage de programmation recommandé en priorité")
    st.pyplot(fig13)
    #Q25/Q38
    
    
    #'activités qui constituent une partie importante de votre rôle au travail'
    data_q23 = data[[i for i in data.columns if 'Q23' in i]]
    data_q23_count = pd.Series(dtype='int')
    for i in data_q23.columns:
        data_q23_count[data_q23[i].value_counts().index[0]] = data_q23[i].count()
    data_q23_count.unique()
    fig8, ax = plt.subplots(1,1, figsize=(20, 20))
    ax.set_xticklabels(data_q23_count.index, fontfamily='serif', rotation=40)
    #fig8.text(0.13, 0.95, 'Visualization Library', fontsize=15, fontweight='bold', fontfamily='serif')
    ax.grid(axis='y', linestyle='-', alpha=0.4)
    #plt.show()
    total=data_q23_count.sum()
    percent = data_q23_count/total*100

    new_labels = [i+'  {:.2f}%'.format(j) for i, j in zip(data_q23_count.index, percent)]

    plt.barh(data_q23_count.index, data_q23_count, color=sns.color_palette("crest", 7),  edgecolor='darkgray')
    plt.yticks(range(len(data_q23_count.index)), new_labels, va="center" ,fontsize=14)
    plt.tight_layout()

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.axes.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", left=False)

    st.write("##### Activités qui constituent une partie importante de votre rôle au travail")
    st.pyplot(fig8)
    st.subheader("Dataviz Power BI")
    st.image("Graph1.png")
    st.image("Graph2.png")
    st.image("Graph3.png")
      
elif page == pages[3]: 
    st.write("### Dataviz Power BI")  
    st.image("Graph1.png")
    st.image("Graph2.png")
    st.image("Graph3.png")
    
elif page == pages[4]:  
    st.write("### Jeu des données après nettoyage")     
    #st.write("Décrire le dataset: Se concentrer sur les 3 catégories")
    st.write("- Préparation des données : méthodologie")
    st.write("  1. suppression des colonnes inutiles et des doublons")
    st.write("  2. renommage des colonnes")
    st.write("  3. filtrage du dataframe sur les métiers cibles")
    st.write("  4. gestion des valeurs manquantes")
    st.write("  5. rédecoupage des variables 'code expérience' et 'ML expérience'en 4 catégories")
    st.write("  6. séparation des variables explicatives et de la variable cible")
    st.write("  7. imputation des valeurs manquantes")
    st.write("  8. encodage des variables avec OneHotEncoder et LabelEncoder")
   # st.write("- Dataframe avec que des données catégorielles --> PCA pas trop efficace")
    
    target = df_prep["title"]
    feats= df_prep.drop("title", axis=1)
    
    #séparation des données en jeu entraînement et jeu test
    from sklearn.model_selection import train_test_split

    # Séparation en jeu de train (60%) et jeu de test (40%)
    X_train, X_rest, y_train, y_rest = train_test_split(feats, target, test_size=0.4, random_state=42)

  
    #remplacer les nan des colonnes catégorielles
    from sklearn.impute import SimpleImputer
    cat = ["education", "code_experience", "langage_reco", "plateforme","ML_experience"]
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    X_train[cat] = imputer.fit_transform(X_train[cat])
    df_prep[cat] = imputer.fit_transform(df_prep[cat])
    #df_prep.to_csv("df_prep.csv", index=False)
    st.dataframe(X_train.head())
    #st.dataframe(feats.head())
    st.write("Dimensions du dataframe :")
    st.write(X_train.shape)
    if st.checkbox("Afficher les valeurs manquantes") :
          # Calculer le taux de valeurs manquantes
           missing_ratio = X_train.isna().mean()
           st.dataframe(missing_ratio)
    if st.checkbox("Afficher les doublons") :
            st.write(X_train.duplicated().sum())

                
elif page == pages[5]:

        st.write("### Modélisation")

        target = df_prep["title"]

        feats= df_prep.drop("title", axis=1)

       

        #séparation des données en jeu entraînement et jeu test
        from sklearn.model_selection import train_test_split

        # Séparation en jeu de train (60%) et jeu de test (40%)
        X_train, X_rest, y_train, y_rest = train_test_split(feats, target, test_size=0.4, random_state=42)

        # Séparation du jeu de test en jeu de validation (20%) et jeu de test (20%)
        X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=42)

        
        #remplacer les nan des colonnes catégorielles
        from sklearn.impute import SimpleImputer
        cat = ["education", "code_experience", "langage_reco", "plateforme","ML_experience"]
        imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        X_train[cat] = imputer.fit_transform(X_train[cat])
        X_test[cat]= imputer.transform(X_test[cat])
        X_val[cat]= imputer.transform(X_val[cat])
        
        #encodage de la valeur cible

        #importations
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import LabelEncoder
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.metrics import f1_score
        from sklearn.metrics import confusion_matrix
        from imblearn.over_sampling import RandomOverSampler
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        import joblib
        from sklearn.metrics import r2_score
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from xgboost import XGBClassifier, plot_importance

        from sklearn.neighbors import KNeighborsClassifier

        # Transformer la série en un DataFrame
        y_train_df = y_train.to_frame(name='title')
        y_test_df = y_test.to_frame(name='title')
        y_val_df = y_val.to_frame(name='title')

        # Initialiser LabelEncoder
        le = LabelEncoder()

        # Appliquer l'encodage à la colonne 'title'
        y_train_encoded = le.fit_transform(y_train_df[['title']])
        y_test_encoded = le.transform(y_test_df[['title']])
        y_val_encoded = le.transform(y_val_df[['title']])

        # Créer un DataFrame à partir du résultat encodé
        y_train = pd.DataFrame(y_train_encoded)
        y_test = pd.DataFrame(y_test_encoded)
        y_val = pd.DataFrame(y_val_encoded)

        #y_train

        #Encodage langage reco et plateforme en onehot
        cat = ['langage_reco', 'plateforme']

        # Initialiser OneHotEncoder avec drop='first' pour éviter la redondance avec la première colonne
        oneh = OneHotEncoder(drop='first', sparse=False)

        # Appliquer l'encodage one-hot à cat pour les données d'entraînement et de test
        X_train_encoded = oneh.fit_transform(X_train[cat])
        X_test_encoded = oneh.transform(X_test[cat])
        X_val_encoded = oneh.transform(X_val[cat])

        # Créer des DataFrames à partir des tableaux one-hot encodés
        X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=oneh.get_feature_names_out(cat), index=X_train.index)
        X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=oneh.get_feature_names_out(cat), index=X_test.index)
        X_val_encoded_df = pd.DataFrame(X_val_encoded, columns=oneh.get_feature_names_out(cat), index=X_val.index)

        # Supprimer les colonnes d'origine à partir des DataFrames d'entraînement et de test
        X_train.drop(columns=cat, inplace=True)
        X_test.drop(columns=cat, inplace=True)
        X_val.drop(columns=cat, inplace=True)

        # Concaténer les DataFrames encodés avec les DataFrames d'origine
        X_train = pd.concat([X_train, X_train_encoded_df], axis=1, ignore_index=False)
        X_test = pd.concat([X_test, X_test_encoded_df], axis=1, ignore_index=False)
        X_val = pd.concat([X_val, X_val_encoded_df], axis=1, ignore_index=False)

       # X_train

        #encodage des variables restantes code expérience et ML expérience en label encoder manuel (garder l'ordre)
        def replace_experience(x):
            if x == 'none':
                return 0
            if x == 'moins de 3':
                return 1
            if x == 'de 3 à 10':
                return 2
            if x == 'plus de 10':
                return 3

        X_train['code_experience'] = X_train['code_experience'].apply(replace_experience)
        X_train['ML_experience'] = X_train['ML_experience'].apply(replace_experience)
        X_test['code_experience'] = X_test['code_experience'].apply(replace_experience)
        X_test['ML_experience'] = X_test['ML_experience'].apply(replace_experience)
        X_val['code_experience'] = X_val['code_experience'].apply(replace_experience)
        X_val['ML_experience'] = X_val['ML_experience'].apply(replace_experience)

       # X_train

        #encodage de la variable education en label encoder manuel (garder l'ordre)
        def replace_education(x):
            if x == 'No formal education past high school':
                return 0
            if x == 'Some college/university study without earning a bachelor’s degree':
                return 1
            if x == 'Bachelor’s degree':
                return 2
            if x == 'Master’s degree':
                return 3
            if x == 'Doctoral degree':
                return 4
            if x == 'Professional degree':
                return 3

        X_train['education'] = X_train['education'].apply(replace_education)
        X_test['education'] = X_test['education'].apply(replace_education)
        X_val['education'] = X_val['education'].apply(replace_education)

        for col in X_train.select_dtypes(include=['object']).columns:
            X_train[col] = X_train[col].astype(int)
        for col in X_test.select_dtypes(include=['object']).columns:
            X_test[col] = X_test[col].astype(int)
        for col in X_val.select_dtypes(include=['object']).columns:
            X_val[col] = X_val[col].astype(int)

        #Undersampling
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.metrics import classification_report_imbalanced, geometric_mean_score
        #Random Undersampling
        rUs = RandomUnderSampler()
        X_ru, y_ru = rUs.fit_resample(X_train, y_train)      

    
#X_train

       # y_train

        #X_test

        #X_train.dtypes

        #for col in X_train.select_dtypes(include=['object']).columns:
           # X_train[col] = X_train[col].astype(int)
        #for col in X_test.select_dtypes(include=['object']).columns:
           # X_test[col] = X_test[col].astype(int)
        #for col in X_val.select_dtypes(include=['object']).columns:
            #X_val[col] = X_val[col].astype(int)

        #scaler = StandardScaler()

       # num = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

        #X_train[num] = scaler.fit_transform(X_train[num])

        #X_test[num] = scaler.transform(X_test[num])



        def prediction(classifier):
            if classifier == 'Random Forest':
                clf = RandomForestClassifier()
            elif classifier == 'SVC':
                clf = SVC()
            elif classifier == 'KNN':
                clf = KNeighborsClassifier()
            clf.fit(X_train, y_train)
            return clf
        
        def scores(clf, choice):
            if choice == 'Accuracy':
                return clf.score(X_val, y_val)
            elif choice == 'Confusion matrix':
                return confusion_matrix(y_val, clf.predict(X_val))
            elif choice == 'F1 score':
                labels = [0, 1, 2]
                return f1_score(y_val, clf.predict(X_val),average=None,labels=labels)
            elif choice == 'F1 score moyen':
                labels = [0, 1, 2]
                return f1_score(y_val, clf.predict(X_val),average='weighted', labels=labels)
        
        choix = ['Random Forest', 'SVC', 'KNN','XGBOOST']
        option = st.selectbox('Choix du modèle', choix)
        st.write('Le modèle choisi est :', option)


        if option!='XGBOOST':
            clf = prediction(option)
        else:
            # create model instance
            bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective="binary:logistic")
            bst.fit(X_train, y_train)
            clf=bst
            
            
        display = st.radio('Que souhaitez-vous montrer ?', ('Accuracy', 'Confusion matrix','F1 score', 'F1 score moyen'))
        if display == 'Accuracy':
            st.write(scores(clf, display))
        elif display == 'Confusion matrix':
            st.dataframe(scores(clf, display))
        elif display == 'F1 score':
            st.dataframe(scores(clf, display))
        elif display == 'F1 score moyen':
            st.write(scores(clf, display))

elif page == pages[6]:       
    st.write("### Conclusion")
    st.write('- Base de données catégorielle avec de nombreuses colonnes')
    st.write("- Dataviz nous a permis de faire ressortir quelques tendances sur les 3 métiers considérés")
    st.write("- Travail conséquent sur le nettoyage et l'optimisation de la base de données")
    st.write('- Modélisation : le meilleur modèle est SVM sans application de PCA, sans rééquilibrage des données. Toutefois, la classe 1 des Data Engineer reste mal prédite.')
    st.write("#### Perspectives")
    st.write('- Ajouter des données des autres sondages pour voir si ça améliore notre modèle pour aboutir à la mise en place d un système de recommandation')
    st.write("- À l'avenir, d'autres techniques de prétraitement ou l'exploration d'autres algorithmes de machine learning pourraient être envisagées pour améliorer davantage les performances.")