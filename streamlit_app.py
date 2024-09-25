import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gdown
import joblib

file_id = '11a22bhGoroqqg3ItPVlBfWzcvBGwpWsd'
url = f'https://drive.google.com/uc?id={file_id}'

# Télécharger le fichier CSV dans un DataFrame pandas
@st.cache_data
def load_data():
    gdown.download(url, 'temp_data.csv', quiet=True)
    return pd.read_csv('temp_data.csv')

df = load_data()

#url = 'https://drive.google.com/uc?export=download&id=11a22bhGoroqqg3ItPVlBfWzcvBGwpWsd'
url2 = 'https://drive.google.com/uc?export=download&id=1JV1ynOg0ek-bHJUo8tn7lbCrlwzprhk2'
url3 = 'https://drive.google.com/uc?export=download&id=1OUxAXN2pXLvKbv72idauAgNHmwl3_-YY'

#Mise en place données utiles
#df=pd.read_csv(url)
df_prep=pd.read_csv(url2)
df_prep2 = pd.read_csv(url3)

#Croisement des données par jour, heure et jour semaine afin de les réutiliser dans des graphiques
comptage_jour = df.groupby('Date_comptage').agg({'Comptage_h': 'mean','prcp': 'mean', 'tavg':'mean'}).reset_index()
comptage_heure = df.groupby('Heure_comptage').agg({'Comptage_h': 'mean'}).reset_index()
comptage_jour_sem=df.groupby('Jour_sem_comptage').agg({'Comptage_h': 'mean','prcp': 'mean', 'tavg':'mean'}).reset_index()

#Détermination relation entre le compte de vélos et les jours travaillés ou non
compte_jourtravaille=df.groupby('jour_travaille').agg({'Comptage_h':'mean'}).reset_index()
compte_jourferie =df.groupby('nom_jour_ferie').agg({'Comptage_h':'mean'}).reset_index()
compte_jourferie=compte_jourferie.sort_values(by = 'Comptage_h')


# CSS pour personnaliser l'apparence
st.markdown("""
    <style>
    .header-banner {
        background-image: url('https://www.assurance-prevention.fr/sites/default/files/styles/simplecrop/public/multiple_uploads/195-velo-ville_1.jpg?itok=Ic1HcQ5O&sc=7f22fe0b0bad05bb1a411349ae81a73f');
        background-size: cover;
        background-position: center;
        height: 150px; /* Ajustez la hauteur selon vos besoins */
    }
    </style>
    <div class="header-banner"></div>
    """, unsafe_allow_html=True)

#https://www.assurance-prevention.fr/sites/default/files/styles/simplecrop/public/multiple_uploads/195-velo-ville_1.jpg?itok=Ic1HcQ5O&sc=7f22fe0b0bad05bb1a411349ae81a73f
st.title("Etude des flux de vélos à Paris")
st.sidebar.title("Sommaire")
pages=["Contexte et présentation des données", "Principaux points de passage", "Saisonnalités du vélo", "Modélisation et prédictions", "Conclusion"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0] : 
  st.header("Introduction")
  st.subheader("Contexte et présentation des jeux de données")
  st.write("Nous sommes un groupe de 4 étudiants du cursus Data Analyst de Data Scientest. Nous avons choisi de travailler sur le thème du trafic de vélos à Paris, pour diverses raisons professionnelles et personnelles (intérêt pour une approche logistique, sensibilité au développement de cette mobilité douce...).")

  st.subheader("Objectifs du projet")
  st.markdown("""
- Cibler les localités des infrastructures (voiries, signalétiques, ...) à rénover/améliorer en priorité
- Dresser une image du type de population à cibler lors de communication d’incitation à la pratique du vélo en ville
- Aider à la gestion des stock au sein des parcs de vélos mis à disposition par la ville
""")
  
  st.image('https://upload.wikimedia.org/wikipedia/commons/d/df/Compteur_V%C3%A9los_Passage_Rue_Rivoli_-_Paris_IV_%28FR75%29_-_2020-09-30_-_3.jpg')
  
  st.subheader("Jeu de données et nettoyage")
  st.write("Le dataset de notre projet contient les données de comptage des compteurs à vélo répartis dans la ville de Paris. La volumétrie est d'environ 1 million de lignes, et ayant relativement peu de variables explicatives à disposition nous avons choisi d'intégrer d'autres sources de données dans notre analyse.")
  st.write("Les données vont d'avril 2023 à mai 2024")

  st.markdown(''':blue[Jeu de données comptage vélos : source opendataparis]''', unsafe_allow_html=True)
  st.write("Il s'agit du principal jeu de notre dataset. Il contient les données de comptage de vélos, à leur passage devant des compteurs répartis dans la ville de Paris. ")
  
  st.markdown(''':blue[Jeu de données vacances scolaires et jours fériés : source data.gouv]''', unsafe_allow_html=True)
  st.write("Nous avons choisi d'ajouter ce jeu à notre dataset pour déterminer si le jour est travaillé ou non.")

  st.markdown(''':blue[Jeu de données météo : source meteostats]''', unsafe_allow_html=True)
  st.write("Ce jeu a permis l'ajout de deux données à notre analyse : la température moyenne et les précipitations totales par jour.")

  if st.checkbox("Afficher le dataframe final après nettoyage"):
    st.dataframe(df_prep.head(10))
  
  st.markdown("Ces objectifs peuvent être atteints en étudiant le compte de vélos, en corrélation avec la localisation des compteurs, les variables de temps (heure/jour/mois) et des données externes comme la météo")

  st.subheader("Variable cible")

  st.markdown("La variable cible de notre projet est le comptage horaire de vélos. En voici la distribution")
  
  fig, ax = plt.subplots()
  sns.boxplot(x='Mois_comptage', y='Comptage_h', data=df, ax=ax)
  ax.set_title('Distribution du comptage horaire par mois')
  st.pyplot(fig)

  st.markdown("Le boxplot a mis en évidence certains outliers.")
  st.markdown("Nous avons choisi de supprimer du dataset 17 lignes qui nous paraissaient aberrantes. Il s'agit des comptages horaires supérieurs à 2000 (soit un vélo toutes les deux secondes).")
  fig1, ax1 = plt.subplots()
  df_prep.boxplot(column='Comptage_h', ax=ax1)
  ax1.set_title('Distribution finale du comptage horaire')
  st.pyplot(fig1)


if page == pages[1] : 
  st.header("Principaux points de passage dans la ville")

  # Sélection année 2024 pour limiter les données
  df_2 = df.loc[(df['Année_comptage']==2024) & (df['Comptage_h']<2000)]

  #Créer la carte interactive avec Plotly
  fig2 = px.scatter_mapbox(df_2, lat="Latitude", lon="Longitude", size="Comptage_h", color="Comptage_h", hover_name="Adresse", zoom=11, center={"lat": 48.8566, "lon": 2.3522})
  fig2.update_layout(mapbox_style="carto-positron")
  fig2.update_layout(margin={"r":0, "t":0, "l":0, "b":0})

  # Afficher la carte dans Streamlit
  st.plotly_chart(fig2)
  st.markdown("<i>La taille et la couleur des points représentent le comptage horaire, pour les données de 2024 </i>", unsafe_allow_html=True)



  st.write("La carte de représentation des orientations n'a pas montré de corrélation particulière entre localisation et orientation. Dans un souci de fluidité nous avons choisi de ne pas l'afficher sur Streamlit")


if page == pages[2] : 
  st.header("Saisonnalités du vélo")

  st.subheader("Saisonnalité temporelle : mois, semaine, jour")

  st.write("Notre dataset montre une forte saisonnalité des flux, et ce sur différentes échelles temporelles. Nous avons choisi d'explorer des tendances avec la visualisation de nos données.")

  # Créer la figure avec plusieurs sous-graphiques
  fig4 = make_subplots(rows=3, cols=1, subplot_titles=('Comptage horaire moyen par jour', 'Comptage horaire moyen par jour de la semaine', 'Comptage horaire moyen par heure'))

  jours_de_la_semaine = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']

  # Ajouter les traces pour chaque sous-graphique
  fig4.add_trace(go.Scatter(x=comptage_jour['Date_comptage'], y=comptage_jour['Comptage_h'], mode='lines+markers'), row=1, col=1)
  fig4.add_trace(go.Bar(x=comptage_jour_sem['Jour_sem_comptage'], y=comptage_jour_sem['Comptage_h']), row=2, col=1)
  fig4.add_trace(go.Scatter(x=comptage_heure['Heure_comptage'], y=comptage_heure['Comptage_h'], mode='lines+markers'), row=3, col=1)

  # Mettre à jour le layout du graphique
  fig4.update_layout(height=900, width=800, title_text="Graphiques de comptage",showlegend=False)
  fig4.update_xaxes(categoryorder='array', categoryarray=jours_de_la_semaine, row=2, col=1)

  # Afficher le graphique dans Streamlit
  st.plotly_chart(fig4)

  st.write("Conclusion : ces graphiques montrent l'existence d'une saisonnalité dans la journée et dans la semaine particulièrement marquées. Elle semble liée à des déplacements professionnels : heures de pointe autour de 9h et 18h, et déplacements plus importants les jours de la semaine que le weekend.")



  st.subheader("Saisonnalité temporelle : impact des jours travaillés")

  st.write("Afin de confirmer nos analyses selon lesquelles les déplacements à vélo sont liés aux déplacements de travail, nous avons visualisé le comptage horaire moyen en fonction des jours travaillés ou non. ")

  fig5 = px.bar(compte_jourtravaille,x='jour_travaille',y='Comptage_h', title='Comptage moyen du nombre de vélos par heure, en jour travaillé ou non')

  # Mise à jour du layout du graphique
  fig5.update_layout(xaxis_title='Jour travaillé?',yaxis_title='Comptage moyen horaire')

  # Afficher le graphique  dans Streamlit
  st.plotly_chart(fig5)
  st.markdown("<i>Jour non travaillé : weekend, jours fériés et jours de vacances scolaires parisiennes (zone C)</i>", unsafe_allow_html=True)


  fig6=px.bar(compte_jourferie,x='nom_jour_ferie',y='Comptage_h')
  fig6.update_layout(title = 'Comptage moyen du nombre de vélos par heure, en jour férié ou non',
                  xaxis_title ='Jour férié?',yaxis_title = 'Comptage moyen horaire')
  st.plotly_chart(fig6)

  st.write("Conclusion : en moyenne, on observe presque 30 vélos de plus par heure sur un jour travaillé plutôt qu'un jour non travaillé.")



  st.subheader("Saisonnalité météorologique : température et précipitations")

  #Recherche corrélation entre comptage et température
  fig7 = px.scatter(comptage_jour, "tavg","Comptage_h",title = 'Corrélation entre comptage horaire moyen de vélos et température')
  fig7.update_layout(xaxis_title = 'Température moyenne journée (°C)',yaxis_title='Comptage moyen journée')
  st.plotly_chart(fig7)

  st.write("Conclusion : le comptage moyen horaire de vélos est corrélé à la température moyenne de la journée. On peut supposer que les températures extrêmes (basses ou hautes) découragent les cyclistes.")

   #Recherche corrélation entre comptage et température
  fig8 = px.scatter(comptage_jour, "prcp","Comptage_h",title = 'Corrélation entre comptage horaire moyen de vélos et précipitations')
  fig8.update_layout(xaxis_title = 'Précipitations totales journée (mm)',yaxis_title='Comptage moyen journée')
  fig8.update_xaxes(range=[0, 20])
  st.plotly_chart(fig8)

  st.write("Conclusion : le comptage moyen horaire de vélos est corrélé à aux précipitations totales de la journée. On peut supposer que la pluie décourage les cyclistes.")

 
if page == pages[3] : 
  st.header("Modélisation et prédictions")

  st.subheader("Preprocessing")
  st.markdown(''':blue[Séparation du jeu de données ]''')
  st.write("***Variable cible et variables explicatives***")
  st.write('Variable cible : comptage_h')
  st.write('Les variables explicatives sont les variables restantes soit les données temporelles, géographiques, de météo et de jour travaillé.')
  st.write("***Jeu de test et jeu d'entraînement***")
  st.write('Nous avons choisi une répartition aléatoire avec un jeu de test comprenant 25% des données. ')
  st.markdown(''':blue[Encodage ]''')
  st.write('Encodage de la variable Orientation grâce à un One Hot Encoder.')
  st.markdown(''':blue[Standardisation ]''')
  st.write('Standardisation de toutes nos données grâce à un Standard Scaler.')

  if st.checkbox("Afficher le dataframe après preprocessing"):
    st.dataframe(df_prep2.head(10))


  


  st.subheader("Modélisation")
  st.write('La variable cible est une variable continue, nous sommes donc face à un problème de type REGRESSION')
  # Import des modèles entraînés et de leurs métriques
  from joblib import load
  loaded_model1 = load('model1.joblib')
  loaded_Score1_train = load('Score1_train.joblib')
  loaded_Score1_test = load('Score1_test.joblib')
  loaded_MAE1 = load('MAE1.joblib')
  #loaded_model2 = load('model2.joblib')
  loaded_MAE2 = load('MAE2.joblib')
  loaded_model2bis = load('model2bis.joblib')
  loaded_MAE2bis = load('MAE2bis.joblib')
  l#oaded_model3 = load('model3.joblib')
  loaded_MAE3 = load('MAE3.joblib')
  #loaded_X_train = load('X_train.joblib')

  # Création d'une fonction qui renvoie la ou les métrique(s)
  def Metriques(option):
    if option == 'Regression linéaire':
      return loaded_Score1_train, loaded_Score1_test, loaded_MAE1
    elif option == 'Arbre de regression':
      return loaded_MAE2
    elif option == 'Arbre de regression après Grid Search':
      return loaded_MAE2bis
    elif option == 'Random Forest':
      return loaded_MAE3

  # Création selectbox
  choix = ['Regression linéaire', 'Arbre de regression', 'Arbre de regression après Grid Search', 'Random Forest']
  option = st.selectbox('Choix du modèle', choix)
  if option == 'Regression linéaire':
    st.write('Score train, score test et MAE : ',Metriques(option))
  else:
    st.write('MAE : ',Metriques(option))

  st.markdown(''':blue[Choix des métriques ] ''')
  st.write("Score ou coefficient de détermination : pertinent dans le cas de la régression linéaire")
  st.write("MAE : métrique la moins sensible aux outliers")

  left_co, cent_co,last_co = st.columns(3)
  with cent_co:
    st.image('https://upload.wikimedia.org/wikipedia/commons/f/f7/Other_general_items_-_Bike_--_Smart-Servier.png')

  st.subheader("Analyse du meilleur modèle")
  st.markdown(''':blue[Régression linéaire ]''')
  st.write('Modèle simple mais performances limitées. Erreur absolue moyenne élevée, incapable de capturer variations complexes du trafic cycliste à Paris.')
  st.markdown(''':blue[Arbre de regression]''')
  st.write('Amélioration significative par rapport à la régression linéaire grâce à la capture des relations non linéaires.')
  st.markdown(''':blue[Random Forest]''')
  st.write('Meilleure précision')

  #st.subheader("Caractéristiques principales")

  #Importance des variable pour le modèle de Random Forest retenu
  #import matplotlib.pyplot as plt
  #import pandas as pd
 # feats = loaded_X_train
 # feat_importances = pd.DataFrame(loaded_model3.feature_importances_, index=feats.columns, columns=["Importance"])
  #feat_importances.sort_values(by='Importance', ascending=False, inplace=True)

  #fig9, ax9 = plt.subplots(figsize=(8, 6))
  #feat_importances.plot(kind='bar', ax = ax9)
 # ax9.set_title('Importance des caractéristiques')
 # ax9.set_xlabel('Caractéristiques')
 # ax9.set_ylabel('Importance')

 # st.pyplot(fig9)  

if page == pages[4] : 
  st.header("Conclusion & perspectives")

  st.subheader("Nos recommandations clefs")

  st.markdown(''':blue[Infrastructures cyclables]''')

  st.markdown("""
- Sécurisation et fluidité : Il est prioritaire de sécuriser les pistes cyclables pendant les heures de pointe et d'investir dans leur fluidité sur les créneaux les plus fréquentés.
- Ciblage   des   touristes  :   Investir   dans   des   infrastructures   cyclables   centrales, également   fréquentées   par   les   touristes,   et   promouvoir   les   vélos   en   libre-service (Vélib') auprès de cette population
""")
  
  st.markdown(''':blue[Gestion des vélos en libre service]''')

  st.markdown("""
- Stock et distribution : Augmenter le nombre de vélos disponibles dans les zones les plus fréquentées, en particulier pendant les créneaux de début et fin de journée.
""")
  
  st.subheader("Synthèse")

  st.write("L'un des principaux apports de notre travail a été de mettre en lumière les variables ayant le plus d'impact sur le trafic cycliste, notamment l'heure de la journée. Nos analyses pourront aider à cibler les zones nécessitant des améliorations d'infrastructures et à optimiser la gestion des vélos en libre-service.")

  st.subheader("Améliorations possibles et futures recherches")

  st.markdown("""
- Capacités techniques : Utiliser des ordinateurs plus performants pour aller plus loin dans l'amélioration des paramètres de nos modèles.
- Détection des outliers : Utiliser des méthodes non supervisées pour une meilleure identification des données aberrantes et améliorer ainsi la précision des modèles.
- Suivi des données futures : Poursuivre le projet avec des données actualisées pour fournir des estimations plus précises.
- Croisement avec les données de Vélib' : Intégrer les données des vélos en libre-service pour des recommandations plus ciblées.
""")
