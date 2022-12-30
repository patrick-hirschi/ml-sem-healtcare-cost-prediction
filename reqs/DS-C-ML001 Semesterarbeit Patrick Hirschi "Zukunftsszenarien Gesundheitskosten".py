#!/usr/bin/env python
# coding: utf-8

# In[914]:


get_ipython().run_cell_magic('html', '', '<style>\ntable {float: left}\n</style>')


# # Semesterarbeit Machine Learning
# 
# ## Dokumenteninformationen
# 
# | Titel <img width=200/>| Schweizer Gesundheitskosten: Ein Blick in die Zukunft |
# |:------------	|:----------------------------------------------------	|
# | Schule     	| Fernfachhochschule Schweiz                  	        |
# | Studiengang 	| Certificate of Advanced Studies in Machine Learning 	|
# | Kennung     	| DS-C-ML001.ML.ZH-Sa-1.PVA.HS22/23                   	|
# | Semester    	| Herbstsemester 2022/23                              	| 
# | Dozent    	| **Ilir Fetai**<br>ilir.fetai@ffhs.ch<br>             	| 
# | Autor    	    | **Patrick Hirschi**<br>Geburtsdatum: 12.01.1990<br>Matrikelnummer: 10-179-026<br>Studierenden-ID: 200768<br>patrick-hirschi@gmx.ch<br>                               	| 

# ## Abstract
# 
# text text text
# 
# ![Alt-Text](./img/Onepager_Semesterarbeit_CAS_Machine_Learning_Patrick_Hirschi.png "Onepager Semesterarbeit")

# ## Ordnerstruktur & Hinweise

# Die Ressourcen für die Arbeit sind in die folgende Ordnerstruktur gegliedert und via github Repository verfügbar:

# **Ordnerstruktur**
#   
# * ml-sem-healthcare-cost-prediction _(git folder)_
#   * data _(all input data files)_
#      * archive _(archived versions of input data)_
#      * profiles _(data profiling html output)_
#          * archive _(archived versions of data profiling html output)_
#   * img _(pictures/images)_
#   * README.md
#   * DS-C-ML001 Semesterarbeit Patrick Hirschi "Zukunftsszenarien Gesundheitskosten".ipynb  _(jupyter notebook)_

# Die virtualenvironment ist nicht Teil des git Repositories. Es wird aber ein requirements.txt File zur Verfügung gestellt, mit allen benutzten Modulen inklusive Versionen.

# ## Datenbeschaffung
# ### Bundesamt für Statistik
# 
# #### Online-Datenbank STAT-TAB
# Das Bundesamt für Statistik führt eine statistische Online-Datenbank (STAT-TAB) für den öffentlichen Zugriff auf Daten der amtlichen Statistik. Die Applikation wurde auf Basis der kostenlosen Software PX-WEB 2017 entwickelt entwickelt. Im Hintergrund sind die Daten in multidimensionalen Cubes abgelegt, was viele Filter-/Slicingmöglichkeiten bietet beim Abfragen der Daten. Ebenso existiert eine REST API, mit welcher man einen programmatischen und automatisierten Zugriff auf die Datensets umsetzen kann. Detaillierte Beschreibungen zu den einzelnen Datencubes und den Abfragemöglichkeiten findet man im [Leitfaden für Online-Datenrecherche (STAT-TAB)](https://dam-api.bfs.admin.ch/hub/api/dam/assets/270926/master "Leitfaden für Online-Datenrecherche (STAT-TAB)").
# 
# #### Asset-Datenbank 
# 
# Das Bundesamt für Statistik führt auch noch eine "Asset and Dissemination" Datenbank. Daten die nicht über die oben beschriebene STAT-TAB Datenbank/API bezogen werden können (z.B. weil sie nicht Teil der multidimensionalen Cubes sind) kann man so beziehen. Eine detaillierte Beschreibung findet man in der [BFS DAM-API Swagger Dokumentation](https://dam-api.bfs.admin.ch/hub/swagger-ui/index.html "BFS DAM-API Swagger Dokumentation").

# #### Nutzungsbedingungen
# 
# Sehr wichtig ist, dass die Nutzungsbedingungen der Datensets beachtet werden. Es gibt verschiedene Kategorien:
#   * **OPEN** (offen für alle Zwecke, Quellenangabe empfohlen)
#   * **OPEN BY** (offen für alle Zwecke, Quellenangabe verpflichtend)
#   * **OPEN ASK** (nicht-kommerziell OK, kommerziell muss erfragt werden beim Datenlieferanten, Quellenangabe empfohlen)
#   * **OPEN BY ASK** (nicht-kommerziell OK, kommerziell muss erfragt werden beim Datenlieferanten, Quellenangabe verpflichtend)
#   
# Sämtliche für dieses Projekt verwendeten Datensets sind in der Kategorie **OPEN BY ASK**. Da es sich bei dieser studentischen Arbeit um einen nicht-kommerziellen Zweck handelt, reicht es also die Quelle der Daten anzugeben, und die Nutzung muss nicht noch separat beim Datenlieferanten erfragt werden. Die URLs für das Laden der Daten beschreiben die genutzten Ressourcen eindeutig.

# ### Modulimport

# In[915]:


# import required modules
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import sys
import os
import datetime
import requests
from requests.exceptions import HTTPError
import pandas as pd
import numpy as np
import webbrowser
import statsmodels.api as sm
import sklearn
from math import sqrt
import pmdarima as pm
#!{sys.executable} -m pip install openpyxl

from pandas_profiling import ProfileReport
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller

# sklearn
from sklearn.experimental import enable_iterative_imputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# skforecast
from skforecast.ForecasterAutoreg import ForecasterAutoreg

# Tensorflow / Keras
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras import Input 
from keras.layers import Bidirectional, LSTM, RepeatVector, Dense, TimeDistributed

# Visualization
import plotly 
import plotly.express as px
import plotly.graph_objects as go

# print versions
print('matplotlib: %s' % matplotlib.__version__)
print('requests: %s' % requests.__version__)
print('pandas: %s' % pd.__version__)
print('numpy: %s' % np.__version__)
print('plotly: %s' % plotly.__version__)
print('sklearn: %s' % sklearn.__version__)
print('Tensorflow/Keras: %s' % keras.__version__)

# log success
print(f'The module import was successful!')


# ### Quelldaten laden

# #### Konfiguration

# Die Daten sind über URLs eindeutig beschrieben, und können so einfach neu heruntergeladen werden. Der Export ist konfigurierbar und kann auch ausgeschalten werden. Bei jedem Export werden die bereits vorhandenen Datenfiles mit einem Zeitstempel versehen und archiviert. Dies garantiert die vollständige Reproduzierbarkeit.
#   
# Wenn zusätzliche Datenfiles von STAT-TAB geladen werden sollen, so können die entsprechenden URLs und Filenamen ganz einfach dazukonfiguriert werden, und der Code fürs Laden der Daten muss nicht weiter angefasst werden.

# In[916]:


# switch to True for reloading data or switch to False if wanting to skip 
# the reload and work with the existing data
download_new_source_data = False
# assign directory
directory = './data'
# get current datetime
datetime_now = datetime.datetime.now()

# data source URLs
data_sources = {
    # alt: 'Gesundheitskosten': 'https://www.pxweb.bfs.admin.ch/sq/7d87e7d5-bc25-489a-898a-ddd98b8cbb6d',
    'Gesundheitskosten': 'https://www.pxweb.bfs.admin.ch/sq/ddc67e6e-b175-42fe-9119-7b35abfd972e',
    'Demographie': 'https://www.pxweb.bfs.admin.ch/sq/b1f9ac16-47f4-4e1e-84a5-33b63a632a8d',
    'Bevölkerungsentwicklung': 'https://www.pxweb.bfs.admin.ch/sq/380ccae0-6b03-464d-90d4-8158b789838a',
    'GesundheitskostenFinanzierung': 'https://www.pxweb.bfs.admin.ch/sq/48e475cd-85e8-41c5-b2db-224937367885',
    'Konsumentenpreise': 'https://dam-api.bfs.admin.ch/hub/api/dam/assets/23664208/master'
}
# target filenames
target_filenames = {
    'Gesundheitskosten': 'gesundheitskosten_2010_bis_2020.csv',
    'Demographie': 'bevoelkerung_1981_bis_2021.csv',
    'Bevölkerungsentwicklung': 'szenarien_bevoelkerung_2019_bis_2070.csv',
    'GesundheitskostenFinanzierung': 'gesundheitskosten_finanzierung_1960_bis_2020.csv',
    'Konsumentenpreise': 'konsumentenpreise_1982_bis_2020.xlsx'
}


# #### Laden der Daten über STAT-TAB pxweb-Ressourcen

# Wenn ein (neues) Laden der Daten konfiguriert wurde, wird in einem ersten Schritt aufgeräumt. Alle vorhandenen Datenfiles im ./data Verzeichnis werden dann ins ./data/archive Verzeichnis verschoben.
#   
# Das python requests Modul ermöglicht es mit nur einer Zeile den entsprechenden Datensatz herunterzuladen. In der Folge wird das neue File im lokalen Filesystem erstellt.
#   
# Ein selektives Neuladen von Files ist nicht vorgesehen, wäre aber mit paar wenigen Code-Anpassungen natürlich möglich.

# In[917]:


# datetime method for logging purposes
def get_current_time_str():
    return datetime.datetime.now().strftime("%H:%M:%S.%f")

# only execute this code if the switch "download_new_source_data" is set to true
if download_new_source_data:
    # iterate over old data files in the directory and archive in ./data/archive folder
    for file in os.listdir(directory):
        # join the filepath information
        fullpath = os.path.join(directory, file)
        # split the filepath into filename and fileextension
        filename = os.path.splitext(file)[0]
        fileextension = os.path.splitext(file)[1]
        # generate a datestring to add to the archived filename
        datestring = datetime_now.strftime("%Y%m%d%H%M%S") + '_'
        # checking if it is a CSV/XLSX file to avoid archiving non-data files
        if os.path.isfile(fullpath) and (fullpath.endswith('.csv') or fullpath.endswith('.xlsx')):
            print(f'{get_current_time_str()}: Found file with path: {fullpath}')
            # generate archive filename
            newpath = directory + '/archive/' + datestring + filename + fileextension
            print(f'{get_current_time_str()}: Archive filepath will be: {newpath}')
            # rename the file and move it directly to the archive directory
            os.rename(fullpath,newpath)
            print(f'{get_current_time_str()}: Successfully archived file {fullpath}'
                  f' to {newpath}!') 
            
    # download latest data files from the Federal Office for Statistics of Switzerland (BfS)
    for key, url in data_sources.items():
        try:
            # access the data file url
            response = requests.get(url, allow_redirects=True)          
            # write content to the target file
            open(os.path.join(directory, target_filenames[key]), 'wb').write(response.content)          
            # get file stats
            file_stats = os.stat(os.path.join(directory, target_filenames[key]))
            # If the response was successful, no Exception will be raised
            response.raise_for_status()
        except HTTPError as http_err:
            # log the details of the HTTP exception (specific catch)
            print(f'{get_current_time_str()}: HTTP error occurred while' 
                  f'downloading data file {key} from {url}: {err}') 
        except Exception as err:
            # log the details of any other exception (generic catch)
            print(f'{get_current_time_str()}: Other error occurred while'
                  f'downloading data file {key} from {url}: {err}')  
        else:
            # download and file write was succesful
            print(f'{get_current_time_str()}: Successfully loaded '
                  f'{file_stats.st_size} bytes from {url} into the data file {target_filenames[key]}!') 
else:
    # source data was intentionally not re-loaded
    print(f'{get_current_time_str()}: '
          f'No data was reloaded from source! If this was not intended, change the switch '
          f'download_new_source_data to True.') 


# ## Datenaufbereitung

# Die Dokumentationen für die Quelldaten kann man zwar auch vom Bundesamt für Statistik beziehen, allerdings soll an dieser Stelle trotzdem ein sorgfältiges Profiling der einzelnen Datensets stattfinden.
#   
# Da die Daten von multidimensionalen Cubes geladen wurden, sind sie schön in klassischen Dimensionen und Fakten aufgeteilt/gegliedert. Für die Datenmodellierung wird an dieser Stelle für jedes Datenset analysiert, was für Spalten es gibt und was für Ausprägungen in diesen Spalten ersichtlich sind. Daraus soll ein Zielmodell abgeleitet werden, was mit den richtigen Qualitätsverbesserungen, Transformationen und Imputationen erreicht werden soll, und schlussendlich als Basis für die Vorhersagen verwendet wird.

# ![Alt-Text](./img/Datenmodellierung_Inputstruktur.png "Struktur der Inputdaten")

# ### Profiling

# Wie das Laden der Daten, ist auch das Profiling konfigurierbar. Die Profiling Reports werden als HTML-Dokumente im lokalen Filesystem abgelegt und nachfolgend im Browser aufgerufen. Theoretisch wäre auch eine inline-Anzeige in jupyter ganz einfach umsetzbar, es hat sich allerdings gezeigt, dass diese das Rendering auf github stören und das jupyter Notebook nachfolgend nicht mehr einfach so im Repository einsehbar ist. Aus diesem Grund werden die Profiling Reports hier als File rausgeschrieben.

# #### Konfiguration

# In[918]:


# set this to True if the data needs to be (re-)profiled
profile_data = True
# set the datestring for the profiling operations
profiling_datestring = datetime_now.strftime("%Y%m%d%H%M%S")


# #### Datenset 1: Gesundheitskosten nach Leistung, Geschlecht und Altersklasse
# Zu dem Datenset gibt es die folgenden Zusatzinformationen vom Bundesamt für Statistik ([Quelle: STAT-TAB Datencube Beschreibung](https://www.pxweb.bfs.admin.ch/pxweb/de/px-x-1405000000_103/-/px-x-1405000000_103.px/ "Quelle: STAT-TAB Datencube Beschreibung")).
# 
# **Kontakt:**
#    * Sektion Gesundheitsversorgung, 058 463 67 00, E-Mail: gesundheit@bfs.admin.ch 
# 
# **Einheit:**
#    * Million Franken; Franken 
# 
# **Metainformation:**
#    * Letzte Änderungen: neuer Cube
#    * Stand der Datenbank: 31.3.2022
#    * Erhebungsperiode: Kalenderjahr
#    * Raumbezug: Schweiz
#    * Datenquelle: Kosten und Finanzierung des Gesundheitswesens (COU)
# 
# **Bemerkungen:**
#    * Provisorische Daten für 2020 
#    
# ##### Import der Daten
# Die Daten sind von der Quelle nicht UTF-8 codiert. Aufgrund der speziellen Sonderzeichen der deutschen und französischen Sprache wurde latin-1 gewählt. Der Import funktioniert dann problemlos.

# In[919]:


df_raw_gk = pd.read_csv(os.path.join(directory, target_filenames['Gesundheitskosten']), 
                        header=0, sep=';', encoding='latin-1')
df_raw_gk


# ##### Profiling des Dataframes mit dem pandas-profiling Package

# In[920]:


# only profile the data file if profiling is configured (check switch "profile_data")
if profile_data:
    
    # set filepath (for new profiling report) and archive filepath (for old profiling report)
    print(f'{get_current_time_str()}: Generating filenames...') 
    fullpath = directory + '/profiles/' + target_filenames['Gesundheitskosten'] + '.html'
    archivepath = directory + '/profiles/archive/' + profiling_datestring + '_' +                      target_filenames['Gesundheitskosten'] + '.html'
    print(f'{get_current_time_str()}: Filepath: {fullpath}') 
    print(f'{get_current_time_str()}: Archive Filepath: {archivepath}') 
    
    # archive old profile reports
    if os.path.isfile(fullpath):
        # rename the file and move it directly to the archive directory
        os.rename(fullpath,archivepath)
        print(f'{get_current_time_str()}: Successfully archived file {fullpath}'
                  f' to {archivepath}!') 
        
    # generate profile report
    print(f'{get_current_time_str()}: Generating profiling report...') 
    report = ProfileReport(df_raw_gk)
    print(f'{get_current_time_str()}: Successfully generated profiling report!') 
    
    # dump the report to a HTML file (jupyter inline rendering breaks the github quickview)
    print(f'{get_current_time_str()}: Dumping profiling report to file...') 
    report.to_file(fullpath)
    print(f'{get_current_time_str()}: Successfully dumped profiling report to file {fullpath}!') 
    
    # open the HTML file in a new browser tab
    print(f'{get_current_time_str()}: Trying to open HTML profiling report {fullpath}'
                  f' in a new browser tab...')
    # bash command to open HTML file because IPython is only called server-side 
    # and tab-opening does not work
    get_ipython().system('open {fullpath}')


# ##### Erkenntnisse
# Der Profiling Report gibt eine schöne Übersicht über die Daten im Dataframe. Daraus lassen sich einige Erkenntnisse ziehen:
#    * Keine Zeilen-Duplikate
#    * Keine Missing Values
#    * 4 kategoriale und 11 numerische Variablen (15 insgesamt)
#    * 378 Beobachtungen/Zeilen
#    * Altersklasse in 5 Jahres Bins (0-5, 6-10, etc.)
#    * Unter den 21 Alerts werden einige Korrelationen gemeldet zwischen den Gesundheitskosten der einzelnen Jahre, aber keine Alerts die eine immediate Korrektur-/Bereinigungsaktion erfordern würden.
# 
# Der Report zeigt, dass das Datenset strukturell und inhaltlich sauber aussieht. Transformationen werden erst nötig, wenn dieses Datenset mit den anderen Datensets zusammen ausgewertet werden soll.

# #### Datenset 2: Demografische Bilanz nach Alter und Kanton
# Zu dem Datenset gibt es die folgenden Zusatzinformationen vom Bundesamt für Statistik ([Quelle: STAT-TAB Datencube Beschreibung](https://www.pxweb.bfs.admin.ch/pxweb/de/px-x-0102020000_104/-/px-x-0102020000_104.px/ "Quelle: STAT-TAB Datencube Beschreibung")).
# 
# **Kontakt:**
#    * Sektion Demografie und Migration, +41 58 463 67 11, E-Mail: info.dem@bfs.admin.ch
# 
# **Einheit:**
#    * Person; Ereignis  
# 
# **Metainformation:**
#    * Letzte Änderungen: Neuer Datensatz (2021)
#    * Stand der Datenbank: August 2022
#    * Erhebungsperiode: 1. Januar - 31. Dezember
#    * Raumbezug: Kantone / 01.01.1997
#    * Datenquelle: 1981-2010 Statistik des jährlichen Bevölkerungsstandes (ESPOP), ab 2011 Statistik der Bevölkerung und der Haushalte (STATPOP)
# 
# Die demografische Bilanz zeigt die Veränderung des Bevölkerungsbestandes aufgrund von natürlichen (Geburten, Todesfälle) und räumlichen (Ein- und Auswanderung bzw. interkantonale Zu- und Wegzüge) Bevölkerungsbewegungen sowie allfälligen statistischen Korrekturen.    
# ESPOP war eine Synthesestatistik, die auf der Statistik der natürlichen Bevölkerungsbewegung (BEVNAT), der Statistik der ausländischen Wohnbevölkerung (PETRA) sowie der Wanderungsstatistik der schweizerischen Wohnbevölkerung basierte. Zudem stützte sie sich auf die Eidgenössischen Volkszählungen (VZ) von 1990 und 2000. ESPOP verwendete die Methode der Bevölkerungsfortschreibung. Dabei wird der Bevölkerungsstand per 31. Dezember eines bestimmten Kalenderjahres ermittelt, indem zum Bestand per 31. Dezember des Vorjahres die Geburten und die Zuwanderungen des jeweiligen Kalenderjahres addiert und die Todesfälle und die Abwanderungen subtrahiert werden. 
# STATPOP entnimmt Bestandes- und Bewegungsdaten aus den Personenregistern des Bundes sowie den harmonisierten Einwohnerregistern der Gemeinden und Kantone und beruht somit auf einem anderen Produktionsverfahren als ESPOP. 
# Die Bevölkerungsbestände per 31. Dezember eines Kalenderjahres und per 1. Januar des folgenden Kalenderjahres sind in folgenden Fällen nicht identisch: (1) Anpassung der Bestandesdaten an die VZ (1990/91 bzw. 2000/01); (2) Umstellung von ESPOP auf STATPOP (2010/11); (3) Gebietsstandänderungen auf Ebene Kanton (1993/94 bzw. 1995/96).
# Wegen der Umstellung von ESPOP auf STATPOP entspricht zudem die Zahl der Todesfälle 2010 nicht deren offizieller Zahl gemäss BEVNAT.    
# Die Bezugsbevölkerung der demografischen Bilanz ist die «ständige Wohnbevölkerung», zu der bis 2010 alle schweizerischen Staatsangehörigen mit einem Hauptwohnsitz in der Schweiz, sowie alle ausländischen Staatsangehörigen mit einer Anwesenheitsbewilligung für mindestens 12 Monate gehören. Mit der Einführung von STATPOP wurde die Bezugsbevölkerung neu definiert. Im Vergleich zu früher umfasst sie seit dem 1.1.2011 zusätzlich Personen im Asylprozess mit einer Gesamtaufenthaltsdauer von mindestens 12 Monaten. 
# 
# **Bemerkungen:**
#    * keine relevanten
#    
# ##### Import der Daten
# Die Daten sind von der Quelle nicht UTF-8 codiert. Aufgrund der speziellen Sonderzeichen der deutschen und französischen Sprache wurde latin-1 gewählt. Der Import funktioniert dann problemlos.

# In[921]:


df_raw_dg = pd.read_csv(os.path.join(directory, target_filenames['Demographie']), 
                        header=0, sep=';', encoding='latin-1')
df_raw_dg


# In[922]:


# only profile the data file if profiling is configured (check switch "profile_data")
if profile_data:
    
    # set filepath (for new profiling report) and archive filepath (for old profiling report)
    print(f'{get_current_time_str()}: Generating filenames...') 
    fullpath = directory + '/profiles/' + target_filenames['Demographie'] + '.html'
    archivepath = directory + '/profiles/archive/' + profiling_datestring + '_' +                      target_filenames['Demographie'] + '.html'
    print(f'{get_current_time_str()}: Filepath: {fullpath}') 
    print(f'{get_current_time_str()}: Archive Filepath: {archivepath}') 
    
    # archive old profile reports
    if os.path.isfile(fullpath):
        # rename the file and move it directly to the archive directory
        os.rename(fullpath,archivepath)
        print(f'{get_current_time_str()}: Successfully archived file {fullpath}'
                  f' to {archivepath}!') 
        
    # generate profile report
    print(f'{get_current_time_str()}: Generating profiling report...') 
    report = ProfileReport(df_raw_dg)
    print(f'{get_current_time_str()}: Successfully generated profiling report!') 
    
    # dump the report to a HTML file (jupyter inline rendering breaks the github quickview)
    print(f'{get_current_time_str()}: Dumping profiling report to file...') 
    report.to_file(fullpath)
    print(f'{get_current_time_str()}: Successfully dumped profiling report to file {fullpath}!') 
    
    # open the HTML file in a new browser tab
    print(f'{get_current_time_str()}: Trying to open HTML profiling report {fullpath}'
                  f' in a new browser tab...')
    # bash command to open HTML file because IPython is only called server-side 
    # and tab-opening does not work
    get_ipython().system('open {fullpath}')


# ##### Erkenntnisse
# Der Profiling Report zeigt folgende Erkenntnisse:
#    * Keine Zeilen-Duplikate
#    * Keine Missing Values
#    * 4 kategoriale und 2 numerische Variablen (6 insgesamt)
#    * 8'282 Beobachtungen/Zeilen
#    * Geschlecht und Alter sind uniform verteilt
#    * Unter den 10 Alerts:
#        - Spalte "Kanton" mit konstantem Wert "Schweiz" (Massnahme: Spalte löschen)
#        - Spalte "Staatsangehörigkeit (Kategorie)" mit konstantem Wert "Staatsangehörigkeit (Kategorie) - Total" (Massnahme: Spalte umbenennen und Wert ändern/kürzen auf "Total")
#        - Spalte "Alter" hat 101 verschiedene Wert, 0-99 und mehr Jahre, und einer für "Alter - Total" (Massnahme: Alter zusammenfassen zu Altersklassen in 5 Jahres Bins - 0-5, 6-10, etc.)

# #### Datenset 3: Szenarien zur Bevölkerungsentwicklung der Schweiz 2020-2070 - Bevölkerung und Bewegungen nach Szenario-Variante, Staatsangehörigkeit (Kategorie), Geschlecht, Altersklasse, Jahr und Beobachtungseinheit
# Zu dem Datenset gibt es die folgenden Zusatzinformationen vom Bundesamt für Statistik ([Quelle: STAT-TAB Datencube Beschreibung](https://www.pxweb.bfs.admin.ch/pxweb/de/px-x-0104000000_102/-/px-x-0104000000_102.px/table/tableViewLayout2/ "Quelle: STAT-TAB Datencube Beschreibung")). Ebenso gibt es eine Beschreibung der getroffenen Annahmen ([Quelle: Bundesamt für Statistik](https://www.bfs.admin.ch/bfs/de/home/statistiken/bevoelkerung/zukuenftige-entwicklung/schweiz-szenarien.html "Quelle: Bundesamt für Statistik")).
# 
# **Kontakt:**
#    * Sektion Demografie und Migration, +41 58 463 67 11, E-Mail: info.dem@bfs.admin.ch
# 
# **Einheit:**
#    * Person  
# 
# **Metainformation:**
#    * Letzte Änderungen: neuer Datensatz (2019-2070)
#    * Stand der Datenbank: Mai 2020
#    * Erhebungsstichtag / Erhebungsperiode: 31. Dezember / 2019-2070
#    * Raumbezug: Schweiz und Kantone / 01.01.1997
#    * Datenquelle: SZENARIEN Bevölkerungsszenarien
#    * Detaillierte Informationen: über die Szenarien zur Bevölkerungsentwicklung der Schweiz
#    
# 
# **Bemerkungen:**
#    * keine relevanten
#    
# ##### Import der Daten
# Die Daten sind von der Quelle nicht UTF-8 codiert. Aufgrund der speziellen Sonderzeichen der deutschen und französischen Sprache wurde latin-1 gewählt. Der Import funktioniert dann problemlos.

# In[923]:


df_raw_be = pd.read_csv(os.path.join(directory, target_filenames['Bevölkerungsentwicklung']), 
                        header=0, sep=';', encoding='latin-1')


# In[924]:


# only profile the data file if profiling is configured (check switch "profile_data")
if profile_data:
    
    # set filepath (for new profiling report) and archive filepath (for old profiling report)
    print(f'{get_current_time_str()}: Generating filenames...') 
    fullpath = directory + '/profiles/' + target_filenames['Bevölkerungsentwicklung'] + '.html'
    archivepath = directory + '/profiles/archive/' + profiling_datestring + '_' +                      target_filenames['Bevölkerungsentwicklung'] + '.html'
    print(f'{get_current_time_str()}: Filepath: {fullpath}') 
    print(f'{get_current_time_str()}: Archive Filepath: {archivepath}') 
    
    # archive old profile reports
    if os.path.isfile(fullpath):
        # rename the file and move it directly to the archive directory
        os.rename(fullpath,archivepath)
        print(f'{get_current_time_str()}: Successfully archived file {fullpath}'
                  f' to {archivepath}!') 
        
    # generate profile report
    print(f'{get_current_time_str()}: Generating profiling report...') 
    report = ProfileReport(df_raw_be)
    print(f'{get_current_time_str()}: Successfully generated profiling report!') 
    
    # dump the report to a HTML file (jupyter inline rendering breaks the github quickview)
    print(f'{get_current_time_str()}: Dumping profiling report to file...') 
    report.to_file(fullpath)
    print(f'{get_current_time_str()}: Successfully dumped profiling report to file {fullpath}!') 
    
    # open the HTML file in a new browser tab
    print(f'{get_current_time_str()}: Trying to open HTML profiling report {fullpath}'
                  f' in a new browser tab...')
    # bash command to open HTML file because IPython is only called server-side 
    # and tab-opening does not work
    get_ipython().system('open {fullpath}')


# ##### Erkenntnisse
# Der Profiling Report zeigt folgende Erkenntnisse:
#    * Keine Zeilen-Duplikate
#    * Keine Missing Values
#    * 4 kategoriale und 2 numerische Variablen (6 insgesamt)
#    * 228'384 Beobachtungen/Zeilen
#    * Geschlecht und Alter sind uniform verteilt
#    * zwölf verschiedene Szenario Varianten (Massnahme: aufsplitten in einzelne Dataframes)
#    * Unter den 11 Alerts:
#        - Spalte "Staatsangehörigkeit (Kategorie)" mit konstantem Wert "Staatsangehörigkeit (Kategorie) - Total"(Massnahme: Spalte umbenennen und Wert ändern/kürzen auf "total")
#        - Spalte "Alter" hat 122 verschiedene Werte, 0-120 und mehr Jahre, und einer für "Alter - Total" (Massnahme: Alter zusammenfassen zu Altersklassen in 5 Jahres Bins - 0-5, 6-10, etc.)
#        - "Bevölkerungsstand am 31. Dezember" hat 22505 0-Werte (nicht fehlend, sondern 0). Wenn man sich diese genauer anschaut, macht es aber auch Sinn. Da die Altersskala bis 120 Jahre geht, ist es natürlich möglich, dass es in den Jahren > 100 teilweise 0 Personen gibt, die dieses Alter erreicht haben. Es werden hier also keine weiteren Massnahmen eingeleitet.

# #### Datenset 4: LIK, Totalindex auf allen Indexbasen 1984-2020
# Zu dem Datenset gibt es die folgenden Zusatzinformationen vom Bundesamt für Statistik ([BfS Katalog Datenbank](https://www.bfs.admin.ch/bfs/de/home/statistiken/kataloge-datenbanken/tabellen.assetdetail.23664208.html "Quelle: BfS Katalog Datenbank")).
# 
# **Kontakt:**
#    * Bundesamt für Statistik, +41 58 463 60 11
# 
# **Einheit:**
#    * Prozent 
# 
# **Metainformation:**
#    * Stand der Datenbank:  	03.11.2022
#    * Erhebungsperiode: 1.12.1982-31.10.2022
#    * Datenquelle: Landesindex der Konsumentenpreise
# 
# **Bemerkungen:**
#    * keine relevanten
#    
# ##### Import der Daten

# In[925]:


df_raw_kp = pd.read_excel(os.path.join(directory, target_filenames['Konsumentenpreise']), 'VAR_y-1', header=3)
df_raw_kp


# In[926]:


# only profile the data file if profiling is configured (check switch "profile_data")
if profile_data:
    
    # set filepath (for new profiling report) and archive filepath (for old profiling report)
    print(f'{get_current_time_str()}: Generating filenames...') 
    fullpath = directory + '/profiles/' + target_filenames['Konsumentenpreise'] + '.html'
    archivepath = directory + '/profiles/archive/' + profiling_datestring + '_' +                      target_filenames['Konsumentenpreise'] + '.html'
    print(f'{get_current_time_str()}: Filepath: {fullpath}') 
    print(f'{get_current_time_str()}: Archive Filepath: {archivepath}') 
    
    # archive old profile reports
    if os.path.isfile(fullpath):
        # rename the file and move it directly to the archive directory
        os.rename(fullpath,archivepath)
        print(f'{get_current_time_str()}: Successfully archived file {fullpath}'
                  f' to {archivepath}!') 
        
    # generate profile report
    print(f'{get_current_time_str()}: Generating profiling report...') 
    report = ProfileReport(df_raw_kp)
    print(f'{get_current_time_str()}: Successfully generated profiling report!') 
    
    # dump the report to a HTML file (jupyter inline rendering breaks the github quickview)
    print(f'{get_current_time_str()}: Dumping profiling report to file...') 
    report.to_file(fullpath)
    print(f'{get_current_time_str()}: Successfully dumped profiling report to file {fullpath}!') 
    
    # open the HTML file in a new browser tab
    print(f'{get_current_time_str()}: Trying to open HTML profiling report {fullpath}'
                  f' in a new browser tab...')
    # bash command to open HTML file because IPython is only called server-side 
    # and tab-opening does not work
    get_ipython().system('open {fullpath}')


# ##### Erkenntnisse
# Der Profiling Report zeigt folgende Erkenntnisse:
#    * 2.9% (14) Zeilen-Duplikate
#    * 7.8% (1'989) Missing Values
#    * 39 nicht-supportete, 11 kategoriale und 3 numerische Variablen (53 insgesamt)
#    * 481 Beobachtungen/Zeilen
#    * Unter den 117 Alerts:
#        - Etliche Inkonsistenzen, korrupte Spalten und ähnliches. Bevor hier Verbesserungen angebracht werden, soll allerdings entschieden werden, welche der Zeilen/Spalten überhaupt relevant sind für die Arbeit. Hier wird nur die totale jahresdurchschnittliche Teuerung sowie die Teuerung isoliert für die KAtegorie "Gesundheitspflege" verwendet.

# #### Datenset 5: Kosten und Finanzierung des Gesundheitswesens nach Leistungserbringer, Leistung und Finanzierungsregime (1960-2020)
# Zu dem Datenset gibt es die folgenden Zusatzinformationen vom Bundesamt für Statistik ([BfS Katalog Datenbank](https://www.bfs.admin.ch/bfs/de/home/statistiken/gesundheit/kosten-finanzierung.assetdetail.22324823.html "Quelle: BfS Katalog Datenbank")).
# 
# **Kontakt:**
#    * Bundesamt für Statistik, +41 58 463 60 11
# 
# **Einheit:**
#    * Million Franken  
# 
# **Metainformation:**
#    * Stand der Datenbank:  	31.03.2022
#    * Erhebungsperiode: Kalenderjahr
#    * Datenquelle: Kosten und Finanzierung des Gesundheitswesens (COU) 
# 
# **Bemerkungen:**
#    * Provisorische Daten für 2020 
#    
# ##### Import der Daten

# In[927]:


df_raw_kf = pd.read_csv(os.path.join(directory, target_filenames['GesundheitskostenFinanzierung']), 
                        header=0, sep=';', encoding='latin-1')
df_raw_kf


# In[928]:


# only profile the data file if profiling is configured (check switch "profile_data")
if profile_data:
    
    # set filepath (for new profiling report) and archive filepath (for old profiling report)
    print(f'{get_current_time_str()}: Generating filenames...') 
    fullpath = directory + '/profiles/' + target_filenames['GesundheitskostenFinanzierung'] + '.html'
    archivepath = directory + '/profiles/archive/' + profiling_datestring + '_' +                      target_filenames['GesundheitskostenFinanzierung'] + '.html'
    print(f'{get_current_time_str()}: Filepath: {fullpath}') 
    print(f'{get_current_time_str()}: Archive Filepath: {archivepath}') 
    
    # archive old profile reports
    if os.path.isfile(fullpath):
        # rename the file and move it directly to the archive directory
        os.rename(fullpath,archivepath)
        print(f'{get_current_time_str()}: Successfully archived file {fullpath}'
                  f' to {archivepath}!') 
        
    # generate profile report
    print(f'{get_current_time_str()}: Generating profiling report...') 
    report = ProfileReport(df_raw_kf)
    print(f'{get_current_time_str()}: Successfully generated profiling report!') 
    
    # dump the report to a HTML file (jupyter inline rendering breaks the github quickview)
    print(f'{get_current_time_str()}: Dumping profiling report to file...') 
    report.to_file(fullpath)
    print(f'{get_current_time_str()}: Successfully dumped profiling report to file {fullpath}!') 
    
    # open the HTML file in a new browser tab
    print(f'{get_current_time_str()}: Trying to open HTML profiling report {fullpath}'
                  f' in a new browser tab...')
    # bash command to open HTML file because IPython is only called server-side 
    # and tab-opening does not work
    get_ipython().system('open {fullpath}')


# ##### Erkenntnisse
# Der Profiling Report zeigt folgende Erkenntnisse:
#    * 0.0% (0) Zeilen-Duplikate
#    * 0.0% (0) Missing Values - dies ist allerdings ein Fehler. Im Datensatz sind offensichtlich fehlende Werte mit einem Stern gekennzeichnet. Dies muss bereinigt werden.
#    * Spalte Finanzierungsregime ist obsolet (immer "Finanzierungsregime - Total")
#    * Spalte Leistungserbringer ist obsolet (immer "Leistungserbringer - Total")
#    * Spalte Leistung soll bereinigt werden (Entfernen der Präfixes wie z.B. ">> N")
#    * 38 kategoriale und 26 numerische Variablen (64 insgesamt). Auch hier leidet die Statistik unter den Stern-Werten (Fehlende Werte).
#    * 9 Beobachtungen/Zeilen
#    * Unter den 92 Alerts:
#        - Keine neuen Erkenntnisse.

# ### Bereinigung

# Die im Profiling Abschnitt erwähnten Qualitätsverbesserungen bzw. generell Transformationen (z.B. Aggregationen, Refactoring, Transponieren, etc.) sollen als Vorbereitung auf eine spätere Zusammenführung der einzelnen Datensets durchgeführt werden. Vor jeder Bereinigung wird ein neues Dataframe "geklont" (mit der copy() Methode). So muss bei missglückten Versuchen nicht immer wieder jede Zelle nochmal ausgeführt werden.

# #### Datenset 1: Gesundheitskosten nach Leistung, Geschlecht und Altersklasse 2010-2020

# In[929]:


df_raw_gk.info()


# In[930]:


# copy the dataframe
df_raw_gk_clean = df_raw_gk.copy()

# for the time being no transformations are needed

# check the output after the transformations
df_raw_gk_clean.info()


# Für die weitere verwendung dieses Datensets sollen einige Werte umformatiert werden (String "Jahre" in Altersgruppen entfernen, Totalwerte umbenennen, etc.). Zum Schluss soll das Datenset so pivotiert werden, dass es eine Spalte "Jahr" gibt, wo dann pro Jahr, Service und Altersgruppe die entsprechenden Gesundheitskosten als Zeilen abgebildet sind. Es wird dann also pro Jahr mehrere Zeilen geben. Die Kosten sind in der Einheit "Mio. CHF" aus der Quelle gekommen und werden hier zusätzlich noch mit 1 Mio. multipliziert, damit die Einheit danach "CHF" ist.

# In[931]:


# copy the dataframe
df_raw_gk_clean_category = df_raw_gk_clean.copy()

# filter dataframe and exclude rows with values per capita and month
df_raw_gk_clean_category = df_raw_gk_clean_category.query("Masseinheit!='Wert pro Kopf und Monat'")
# filter dataframe and exclude rows with total per service
df_raw_gk_clean_category = df_raw_gk_clean_category.query("Leistung!='Leistung - Total'")
# filter dataframe and exclude rows with total for genders
df_raw_gk_clean_category = df_raw_gk_clean_category.query("Geschlecht!='Geschlecht - Total'")
# filter dataframe and exclude rows with total for agegroups
df_raw_gk_clean_category = df_raw_gk_clean_category.query("Altersklasse!='Altersklasse - Total'")

# replace strings in age groups with empty strings
df_raw_gk_clean_category['Altersklasse'] = df_raw_gk_clean_category['Altersklasse']                                                 .str.replace(' Jahre','')
df_raw_gk_clean_category['Altersklasse'] = df_raw_gk_clean_category['Altersklasse']                                                 .str.replace(' und mehr Jahre','+')
   
# change column datatype to category
df_raw_gk_clean_category['Altersklasse'] = df_raw_gk_clean_category['Altersklasse'].astype('category')
# rename column "Altersklasse" to "Altersgruppe"
df_raw_gk_clean_category.rename(columns = {'Altersklasse':'Altersgruppe'}, inplace = True)

# pivot and stack the table to have the years in a specific column
df_raw_gk_clean_category = pd.pivot_table(
    df_raw_gk_clean_category,
    values=['2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020'],
    index=['Masseinheit','Leistung','Altersgruppe','Geschlecht']
    ).stack().reset_index()

# rename automatically created columns
df_raw_gk_clean_category.rename(columns = {'level_4':'Jahr', 0:'Kosten'}, inplace = True)

# group and sum over genders
df_raw_gk_clean_category = df_raw_gk_clean_category                                     .groupby(['Jahr','Masseinheit','Leistung','Altersgruppe'])['Kosten']                                     .sum()                                     .reset_index()

# multiply all values in column "Kosten" by 1'000'000 since the unit is "million swiss francs"
df_raw_gk_clean_category['Kosten'] = df_raw_gk_clean_category['Kosten']                                     .mul(1000000)

# drop unnecessary columns
df_raw_gk_clean_category.drop('Masseinheit', axis=1, inplace=True)

# change column type to int
df_raw_gk_clean_category['Jahr'] = df_raw_gk_clean_category['Jahr']                                     .astype(int)

df_raw_gk_clean_category


# Nach allen Transformationen hat Datenset 1 das gewünschte Format.

# #### Datenset 2: Demografische Bilanz nach Alter und Kanton 1981-2021

# In[932]:


df_raw_dg


# Wie bei Datenset 1 sollen auch hier einige Werte umformatiert werden. Datenset 2 ist aktuell in Altersgruppen mit Breite von einem Jahr aufgeteilt. dies soll auf die gewohnten 5-Jahres-Gruppen aggregiert werden.  

# In[933]:


# copy the dataframe
df_raw_dg_clean = df_raw_dg.copy()
# drop unnecessary columns
df_raw_dg_clean.drop('Kanton', axis=1, inplace=True)
df_raw_dg_clean.drop('Staatsangehörigkeit (Kategorie)', axis=1, inplace=True)

# filter dataframe and exclude rows with total count over all age(groups)
df_raw_dg_clean = df_raw_dg_clean.query("Alter!='Alter - Total'")
# remove all non-number characters in column "Alter"
df_raw_dg_clean = df_raw_dg_clean.assign(Alter = lambda x: x['Alter'].str.extract('(\d+)'))
# change column type to int
df_raw_dg_clean['Alter'] = df_raw_dg_clean['Alter'].astype('int')

# generate age groups of 5 years
bins= [0,6,11,16,21,26,31,36,41,46,51,56,61,66,71,76,81,86,91,96,np.inf]

labels = ['0-5','6-10','11-15','16-20','21-25','26-30','31-35','36-40','41-45','46-50',
          '51-55','56-60','61-65','66-70','71-75','76-80','81-85','86-90','91-95','96+']

df_raw_dg_clean['Altersgruppe'] = pd.cut(df_raw_dg_clean['Alter'],                                           bins=bins,                                            labels=labels,                                            right=False)
df_raw_dg_clean['Altersgruppe'] = df_raw_dg_clean['Altersgruppe']                                     .cat                                     .add_categories('unbekannt')                                     .fillna('unbekannt')

# check the output after the transformations
df_raw_dg_clean


# Nachdem im vorhergehenden Schritt jeder Zeile die entsprechende Altersgruppe zugewiesen wurde, kann nun nach Jahr und zugewiesener Gruppe gruppiert und die Anzahl Personen aufsummiert werden. Es resultieren die totalen Zahlen pro Jahr und Altersgruppe.

# In[934]:


# copy the dataset
df_raw_dg_clean_category = df_raw_dg_clean.copy()

# group and sum the data per year and age group
df_raw_dg_clean_category = df_raw_dg_clean_category                                 .groupby(['Jahr', 'Altersgruppe'])['Bestand am 31. Dezember']                                 .sum()                                 .reset_index()

# print the result
df_raw_dg_clean_category


# Eine Pivotierung ist hier nicht nötig. Datenset 2 ist somit fertig transformiert.

# #### Datenset 3: Szenarien zur Bevölkerungsentwicklung der Schweiz 2020-2070

# In[935]:


df_raw_be


# Das Datenset beinhaltet viele verschiedene errechnete Bevölkerungsszenarien. Im Rahmen dieser Arbeit können und sollen nicht alle davon untersucht werden. Der primäre Fokus soll auf den Referenzszenarien und auf den Szenarien "verstärkte Alterung" und "höhere Lebenserwartung" geht. Auch hier können aber andere Szenarien dazukonfiguriert werden.

# In[936]:


df_raw_be["Szenario-Variante"].unique()


# Auch hier müssen einige Spalten bereinigt werden und die Daten in Altersgruppen aufgeteilt werden. Da im Rahmen dieser Arbeit das Feature "Geschlecht" nicht berücksichtigt wird, sollen nur Zeilen mit "Geschlecht - Total" behalten werden.

# In[937]:


# copy the dataframe
df_raw_be_clean = df_raw_be.copy()

scenarios_to_keep = [  'Referenzszenario A-00-2020',  '\'hohes\' Szenario B-00-2020',  '\'tiefes\' Szenario C-00-2020',  'Szenario D-00-2020 \'verstärkte Alterung\'', # 'Szenario E-00-2020 \'abgeschwächte Alterung\'', \
# 'Variante A-01-2020 \'höhere Fertilität\'', \
# 'Variante A-02-2020 \'niedrigere Fertilität\'', \
 'Variante A-03-2020 \'höhere Lebenserwartung bei der Geburt\'' \
# 'Variante A-04-2020 \'niedrigere Lebenserwartung bei der Geburt\'', \
# 'Variante A-05-2020 \'hohes Wanderungssaldo\'', \
# 'Variante A-06-2020 \'tiefes Wanderungssaldo\'', \
# 'Variante A-07-2020 \'stabile Auswanderungsziffern\'' \
]

# filter dataframe for specific scenarios only (configured just above)
df_raw_be_clean = df_raw_be_clean[df_raw_be_clean["Szenario-Variante"]                         .isin(scenarios_to_keep)]

# filter dataframe and only keep rows with gender "Geschlecht - Total"
df_raw_be_clean = df_raw_be_clean.query("Geschlecht=='Geschlecht - Total'")
# remove all non-number characters in column "Alter"
df_raw_be_clean = df_raw_be_clean.assign(Alter = lambda x: x['Alter'].str.extract('(\d+)'))

# drop nan age groups for the total counts
df_raw_be_clean.dropna(inplace=True)

# change column type to int
df_raw_be_clean['Alter'] = df_raw_be_clean['Alter'].astype('int')

# generate age groups of 5 years
bins= [0,6,11,16,21,26,31,36,41,46,51,56,61,66,71,76,81,86,91,96,np.inf]

labels = ['0-5','6-10','11-15','16-20','21-25','26-30','31-35','36-40','41-45','46-50',
          '51-55','56-60','61-65','66-70','71-75','76-80','81-85','86-90','91-95','96+']

df_raw_be_clean['Altersgruppe'] = pd.cut(df_raw_be_clean['Alter'],                                           bins=bins,                                            labels=labels,                                            right=False)
df_raw_be_clean['Altersgruppe'] = df_raw_be_clean['Altersgruppe']                                     .cat                                     .add_categories('unbekannt')                                     .fillna('unbekannt')

# drop unnecessary columns
df_raw_be_clean.drop('Staatsangehörigkeit (Kategorie)', axis=1, inplace=True)
df_raw_be_clean.drop('Geschlecht', axis=1, inplace=True)

# check the output after the transformations
df_raw_be_clean


# Nachdem im vorhergehenden Schritt wieder jeder Zeile die entsprechende Altersgruppe zugewiesen wurde, kann nun auch hier nach Jahr und zugewiesener Gruppe gruppiert und die Anzahl Personen aufsummiert werden. Pivotiert resultieren die totalen Zahlen pro Jahr, Szenario und Altersgruppe.

# In[938]:


# copy the dataset
df_raw_be_clean_category = df_raw_be_clean.copy()
# group and sum the data per year and age group
df_raw_be_clean_category = df_raw_be_clean                                 .groupby(['Jahr', 'Altersgruppe', 'Szenario-Variante'])['Bevölkerungsstand am 31. Dezember']                                 .sum()                                 .reset_index()

# rename column to ensure integrity
df_raw_be_clean_category.rename(columns = {'Bevölkerungsstand am 31. Dezember':'Bestand am 31. Dezember'}, inplace = True)

# pivot and stack the table to have the years in a specific column
df_raw_be_pivoted = pd.pivot_table(
    df_raw_be_clean_category,
    values='Bestand am 31. Dezember',
    index=['Jahr','Szenario-Variante'],
    columns=['Altersgruppe']
    ).reset_index()

# drop the column for age group "unbekannt" since it is always 0
df_raw_be_pivoted.drop('unbekannt', axis=1, inplace=True)

# transform pivoted data frame to regular one
df_raw_be_pivoted = pd.DataFrame(df_raw_be_pivoted.to_records())

# remove the unnecessary index column
df_raw_be_pivoted.drop('index', axis=1, inplace=True)

# print the result
df_raw_be_pivoted


# #### Datenset 4: Konsumentenpreise: Jahresdurchschnittliche Teuerung 1984-2021

# In[939]:


df_raw_kp.info()


# Anders als bei den anderen Datensets war hier die Quelle ein Excel File. Glücklicherweise brauchen wir daraus nur zwei Zeilen, in welchen die Werte für den totalen Landesindex der Konsumentenpreise drin ist, sowie der Index gefiltert auf "Gesundheitspflege". Die Werte pro Jahr sind in Spalten abgelegt. 
#   
# Um die spätere Zusammenführung der Daten mit den anderen datensets zu erleichtern, soll auch hier eine Pivotierung gemacht werden, damit es eine Spalte "Jahr" gibt.

# In[940]:


# copy the dataframe
df_raw_kp_clean = df_raw_kp.copy()

# convert all column names to strings and strip whitespace
df_raw_kp_clean = df_raw_kp_clean.rename(columns=lambda x: str(str(x).strip()))

# filter dataframe for row with total inflation (Code 100_100) and inflation 
# for healthcare products (Code 100_6)
df_raw_kp_clean = df_raw_kp_clean.query("(Code=='100_100') or (Code=='100_6')").reset_index()

# rename category column
df_raw_kp_clean.rename(columns={ "Position_D": "Kategorie" }, inplace = True)
# drop unnecessary columns
df_raw_kp_clean = df_raw_kp_clean[['Kategorie','1983', '1984', '1985', '1986', '1987', '1988', 
                                   '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', 
                                   '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', 
                                   '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', 
                                   '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', 
                                   '2021']]

# drop duplicates
df_raw_kp_clean.drop_duplicates(keep='first', inplace=True)

# drop NaN column
df_raw_kp_clean.drop('1983', axis=1, inplace=True)

# change all columns except "Kategorie" to float data type
df_raw_kp_clean.iloc[:,1:] = df_raw_kp_clean.iloc[:,1:].astype(float)

# pivot and stack the table to have the years in a specific column
df_raw_kp_clean = pd.pivot_table(
    df_raw_kp_clean,
    values=df_raw_kp_clean.iloc[:,1:],
    index=['Kategorie']
    ).stack().reset_index()

# rename automatically created columns
df_raw_kp_clean.rename(columns = {'level_1':'Jahr', 0:'LIK'}, inplace = True)

# print the result
df_raw_kp_clean


# Datenset 4 ist somit fertig bereinigt.

# #### Datenset 5: Kosten und Finanzierung des Gesundheitswesens nach Leistungserbringer, Leistung und Finanzierungsregime (1960-2020)

# In[941]:


df_raw_kf


# Datenset 5 besteht aus vielen Stern-Werten, welche als placeholder für fehlende Werte eingesetzt wurden. Daneben gibt es auch hier Spalten, die weggelassen werden sollen (Leistungserbringer & Finanzierungsregime), wie auch Spalten wo die Werte umformatiert werden sollen (Leistung). Um die Einheit "Mio. CHF" in "CHF" umzuwandeln werden die Kosten mit einer Mio. multipliziert und nach der Pivotierung ist dann auch Datenset 5 fertig bereinigt.

# In[942]:


# copy the dataframe
df_raw_kf_clean = df_raw_kf.copy()
# drop unnecessary columns
df_raw_kf_clean.drop('Leistungserbringer', axis=1, inplace=True)
df_raw_kf_clean.drop('Finanzierungsregime', axis=1, inplace=True)
# filter out total values
# df_raw_kf_clean = df_raw_kf_clean.query("Leistung!='Leistung - Total'")

# clean values in column "Leistung"
df_raw_kf_clean['Leistung'] = df_raw_kf_clean['Leistung']                                 .replace(['Leistung - Total'],                                          'LeistungenTotal')
df_raw_kf_clean['Leistung'] = df_raw_kf_clean['Leistung']                                 .replace(['>> L Stationäre Kurativbehandlung'],                                          'Stationäre Kurativbehandlung')
df_raw_kf_clean['Leistung'] = df_raw_kf_clean['Leistung']                                 .replace(['>> M Ambulante Kurativbehandlung'],                                          'Ambulante Kurativbehandlung')
df_raw_kf_clean['Leistung'] = df_raw_kf_clean['Leistung']                                 .replace(['>> N Rehabilitation'],                                           'Rehabilitation')
df_raw_kf_clean['Leistung'] = df_raw_kf_clean['Leistung']                                 .replace(['>> O Langzeitpflege'],                                           'Langzeitpflege')
df_raw_kf_clean['Leistung'] = df_raw_kf_clean['Leistung']                                 .replace(['>> P Unterstützende Dienstleistungen'],                                           'Unterstützende Dienstleistungen')
df_raw_kf_clean['Leistung'] = df_raw_kf_clean['Leistung']                                 .replace(['>> Q Gesundheitsgüter'],                                           'Gesundheitsgüter')
df_raw_kf_clean['Leistung'] = df_raw_kf_clean['Leistung']                                 .replace(['>> R Prävention'],                                           'Prävention')
df_raw_kf_clean['Leistung'] = df_raw_kf_clean['Leistung']                                 .replace(['>> S Verwaltung'],                                           'Verwaltung')
# clean *-values to nan
df_raw_kf_clean = df_raw_kf_clean.replace(['*'], np.nan)

# change all columns except "Leistung" to float data type
df_raw_kf_clean.iloc[:,1:] = df_raw_kf_clean.iloc[:,1:].astype(float)

# multiply all values by 1'000'000 since the unit is "million swiss francs"
df_raw_kf_clean.iloc[:,1:] = df_raw_kf_clean.iloc[:,1:].mul(1000000)

# pivot and stack the table to have the years in a specific column
df_raw_kf_clean = pd.pivot_table(
    df_raw_kf_clean,
    values=df_raw_kf_clean.iloc[:,1:],
    index=['Leistung']
    ).stack().reset_index()

# rename automatically created columns
df_raw_kf_clean.rename(columns = {'level_1':'Jahr', 0:'Kosten'}, inplace = True)

# change column type to int
df_raw_kf_clean['Jahr'] = df_raw_kf_clean['Jahr'].astype(int)

# check the output after the transformations
df_raw_kf_clean


# ### Sanity Checks

# Theoretisch sollten an dieser Stelle einige automatisierte Testcases ausgeführt werden können um sicherzustellen, dass nach all den Transformationen der Inhalt der Daten weiterhin korrekt ist. Dies ist aber sehr zeitaufwändig und deswegen sollen einfach einige schnelle manuelle Prüfungen gemacht werden.

# #### Kosten pro Jahr und Leistung zwischen 2010 und 2020 (Datenset 1 & 5)

# In[943]:


# group and sum dataset 1 over year and service and compare to other dataset
df_raw_gk_clean_category_sanity_check = df_raw_gk_clean_category.copy()
df_raw_gk_clean_category_sanity_check = df_raw_gk_clean_category_sanity_check                                             .groupby(['Jahr','Leistung'])['Kosten']                                             .sum()                                             .reset_index()
df_raw_gk_clean_category_sanity_check = df_raw_gk_clean_category_sanity_check                                             .query("(Jahr>=2010) and (Jahr<=2020)")                                             .reset_index(drop=True)

# remove entries from 1995 to 2010 from dataset 5
df_raw_kf_clean_sanity_check = df_raw_kf_clean.copy()
df_raw_kf_clean_sanity_check = df_raw_kf_clean_sanity_check                                             .query("(Jahr>=2010) and (Jahr<=2020)")                                             .reset_index(drop=True)

# reindex dataframes to be able to compare
columns_titles = ["Jahr","Leistung","Kosten"]
df_raw_kf_clean_sanity_check.sort_values(by=["Jahr","Leistung"], ascending=False)
df_raw_gk_clean_category_sanity_check.sort_values(by=["Jahr","Leistung"], ascending=False)
df_raw_kf_clean_sanity_check = df_raw_kf_clean_sanity_check                                             .reindex(columns=columns_titles)
df_raw_gk_clean_category_sanity_check = df_raw_gk_clean_category_sanity_check                                             .reindex(columns=columns_titles)

# compare the datasets
merged = pd.merge(df_raw_kf_clean_sanity_check, df_raw_gk_clean_category_sanity_check, 
                  on=['Jahr','Leistung'], how='inner')
merged


# Die Differenzen sind auf die Rundung bei Datenset 1 pro Altersgruppe zurückzuführen. Bei Datenset 5 sind es von Anfang an aggregierte Werte über alle Altersguppen und die Zahl ist somit genauer. Da für dieses Vorhaben aber die Zahlen pro Altersgruppe relevant sind, und keine genauere/komplettere Quelle gefunden wurde, gibt es keine Wahl. Für einen ersten Vorhersageversuch, soll aber mit den totalen Kosten pro Leistung über alle Altersgruppen gearbeitet werden. Eventuell sind dadurch bereits gute Resultate zu erzielen.
# Falls nicht, kann in einem zweiten Versuch noch die Aufteilung auf die Altersklassen miteinbezogen werden.

# #### Demografische Bilanz (Datenset 2)
# Die Bevölkerungsszenarien (Datenset 3) starten im Jahr 2019 und Datenset 2 reicht bis 2021. Somit können wenigstens die Zahlen für 2019, 2020 und 2021 noch kurz geprüft werden.

# In[944]:


# dataset 2: group over year and sum the people count
df_raw_dg_clean_category_sanity_check = df_raw_dg_clean_category.copy()
df_raw_dg_clean_category_sanity_check = df_raw_dg_clean_category_sanity_check                                     .groupby(['Jahr'])['Bestand am 31. Dezember']                                     .sum()                                     .reset_index()
df_raw_dg_clean_category_sanity_check = df_raw_dg_clean_category_sanity_check                                     .query("(Jahr>=2019) and (Jahr<=2021)")                                     .reset_index(drop=True)

# dataset 3: group over year and sum the people count
df_raw_be_clean_sanity_check = df_raw_be_clean.copy()
df_raw_be_clean_sanity_check.rename(columns={"Bevölkerungsstand am 31. Dezember":                                               "Bestand am 31. Dezember" }, inplace = True)
df_raw_be_clean_sanity_check = df_raw_be_clean_sanity_check                                     .query("(`Szenario-Variante`=='Referenzszenario A-00-2020') and                                             (Jahr >= 2019) and                                             (Jahr <= 2021) and                                             (Alter!='Alter - Total')")                                     .reset_index(drop=True)
df_raw_be_clean_sanity_check = df_raw_be_clean_sanity_check                                     .groupby(['Jahr'])['Bestand am 31. Dezember']                                     .sum()                                     .reset_index()

# compare the datasets
merged = pd.merge(df_raw_be_clean_sanity_check, df_raw_dg_clean_category_sanity_check, 
                  on=['Jahr'], how='inner')
merged


# Wenig überraschend gibt es schon in den ersten drei Jahren der vorhergesagten Referenzszenarien leichte Abweichungen zu den effektiv gemessenen Werten. Die Abweichungen bewegen sich zwischen 9'000 und 23'000 Personen, was einem Fehler von 0.26% im Jahr 2021 entspricht. Die folgende Grafik zeigt grafisch die Entwicklung für die drei Referenzszenarien ([Quelle: Bundesamt für Statistik](https://www.bfs.admin.ch/bfs/de/home/statistiken/bevoelkerung/zukuenftige-entwicklung/schweiz-szenarien.html "Quelle: Bundesamt für Statistik")):
# 
# ![Alt-Text](./img/Referenzszenarien_Bevoelkerungsentwicklung.png "Referenzszenarien Bevölkerungsentwicklung")

# #### Szenarien Bevölkerungsentwicklung (Datenset 3)
# Die Berechnungen/Vorhersagen wurden allesamt vom Bundesamt für Statistik selber durchgeführt und eine Prüfung der Angaben ist nicht möglich. Theoretisch könnte man eigene Szenarien entwickeln und die Zahlen des Bundesamtes challengen. Darauf wird abgesehen von dem kurzen Check im vorherigen Abschnitt an dieser Stelle aber verzichtet.

# #### Landesindex der Konsumentenpreise (Datenset 4)

# "Der Landesindex der Konsumentenpreise (LIK) ist ein gesamtschweizerischer Indikator für die Preisentwicklung der für Konsumentinnen und Konsumenten bedeutsamen Waren und Dienstleistungen. Er dient unter anderem als Grundlage für die Geld- und die allgemeine Wirtschaftspolitik, zur Bestimmung des realen Wirtschaftswachstums und der realen Lohn- und Umsatzentwicklung wie auch zur Beurteilung der internationalen Wettbewerbsfähigkeit der Schweiz.
# Der LIK wird monatlich vom Bundesamt für Statistik (BFS) nach dem Inländerkonzept aufgrund von aktuellen Preiserhebungen berechnet. Grundlage für die Indexberechnung bildet der sogenannte Warenkorb. Darin wird definiert, mit welchem prozentualen Gewicht die Preise der einzelnen Waren und Dienstleistungen in die Indexberechnung eingehen. Der Warenkorb bildet die Struktur der Konsumausgaben der privaten Haushalte so realitätsgetreu wie möglich nach; seit Dezember 2001 wird er anhand der Haushaltsbudgeterhebung (HABE) jährlich neu gewichtet. Der Mietpreisindex ist eine Komponente des Konsumentenpreisindex und hat am aktuellen Warenkorb einen Anteil von rund 20 Prozent.
# Der LIK wird seit 1922 berechnet und wurde seither neun Revisionen unterzogen (1926, 1950, 1966, 1977, 1982, 1993, 2000, 2005 und 2010). Mit der letzten Revision 2010 stellte das BFS den LIK auf neue Grundlagen (Basis Dezember 2010 = 100), die ab Januar 2011 für die Berechnung massgebend sind. Wie die früheren Indexerneuerungen passte auch die jüngste Revision den LIK an veränderte Markt- und Konsumstrukturen an und berücksichtigte neue methodische Entwicklungen. Schwerpunkte waren unter anderem die Überarbeitung des Mietpreisindex, Fragen zur Qualitätsbereinigung und der Erhebungstechniken. Der LIK 2010 ist nach wie vor ein Preisindex und kein Lebenshaltungskostenindex.
# Um einen internationalen Vergleich der Teuerung zu ermöglichen, haben die Mitgliedstaaten der EU einen Indikator eingeführt, der anhand einer harmonisierten Methode berechnet wird: den harmonisierten Verbraucherpreisindex (HVPI). Weil der Index als wichtiges Steuerungsinstrument für die Währungspolitik gilt, wird er vom BFS seit 2008 auch für die Schweiz publiziert." [(Quelle: Statistik Amt Luzern)](https://www.lustat.ch/services/lexikon/quellen-und-erhebungen?id=401 "Quelle: Statistik Amt Luzern")

# Mit anderen Worten: Die Zahl wird vom BfS selbst gemessen und herausgegeben. Somit ist kein sinnvoller Vergleich mit anderen Quellen möglich. Der Verbraucherpreisindex der EU-weit genutzt wird ist für diese Arbeit nicht relevant. Es werden keine Vergleiche mit anderen Ländern angestrebt.

# ### Zusammenführung & Modellierung

# In einem ersten Schritt soll die Grundstruktur des zusammengeführten Datensatzes erstellt werden. 
# 
# ![Alt-Text](./img/Datenmodellierung_Transformation.png "Transformation zum Zielmodell")
# 
# Das angestrebte Endziel: Pro Jahr soll es eine Zeile geben und jeweils für das entsprechende Jahr Spalten mit der Anzahl Personen pro Altersgruppe. Dazu kommen dann die Spalten mit den Kosten pro Leistung (Stationär, Ambulant, etc.) im jeweiligen Jahr, wie auch der total LIK und der LIK für Gesundheitsgüter.

# In[945]:


# copy the original data set
df_dg_pivoted = df_raw_dg_clean_category.copy()

# pivot and stack the table to have the years in a specific column
df_dg_pivoted = pd.pivot_table(
    df_raw_dg_clean_category,
    values='Bestand am 31. Dezember',
    index=['Jahr'],
    columns=['Altersgruppe']
    ).reset_index()

# drop the column for age group "unbekannt" since it is always 0
df_dg_pivoted.drop('unbekannt', axis=1, inplace=True)

# transform pivoted data frame to regular one
df_dg_pivoted = pd.DataFrame(df_dg_pivoted.to_records())

# remove the unnecessary index column
df_dg_pivoted.drop('index', axis=1, inplace=True)

df_dg_pivoted.head(5)


# In[946]:


# copy the original data set
df_kf_pivoted = df_raw_kf_clean.copy()

# pivot and stack the table to have the years in a specific column
df_kf_pivoted = pd.pivot_table(
    df_raw_kf_clean,
    values='Kosten',
    index=['Jahr'],
    columns=['Leistung']
    ).reset_index()

# transform pivoted data frame to regular one
df_kf_pivoted = pd.DataFrame(df_kf_pivoted.to_records())

# remove the unnecessary index column
df_kf_pivoted.drop('index', axis=1, inplace=True)

df_kf_pivoted.head(5)


# Die Zusammenführung dieser zwei pivotierten Datensets ist nun ganz einfach mit einem LEFT JOIN über die Spalte "Jahr" zu erreichen.
# 
# Es ist wichtig, hier einen LEFT JOIN zu verwenden und nicht einen INNER JOIN. Wir wollen die Jahre für die keine Angaben für die Gesundheitskosten pro Leistung vorhanden ist, aber sehr wohl Angaben zu den Altersgruppen, auch im Output behalten.

# In[947]:


# merge the datasets
df_dg_kf_merged = pd.merge(df_dg_pivoted, df_kf_pivoted, 
                  on=['Jahr'], how='left')
df_dg_kf_merged.tail(10)


# Jetzt sollen noch die zwei Spalten für den Landesindex der Konsumentenpreise dazugenommen werden.

# In[948]:


df_raw_kp_clean.head(5)


# In[949]:


# copy the original data set
df_kp_pivoted = df_raw_kp_clean.copy()

# pivot and stack the table to have the years in a specific column
df_kp_pivoted = pd.pivot_table(
    df_raw_kp_clean,
    values='LIK',
    index=['Jahr'],
    columns=['Kategorie']
    ).reset_index()

# transform pivoted data frame to regular one
df_kp_pivoted = pd.DataFrame(df_kp_pivoted.to_records())

# remove the unnecessary index column
df_kp_pivoted.drop('index', axis=1, inplace=True)

# rename category column
df_kp_pivoted.rename(columns={"    Gesundheitspflege":"LIK_Gesundheitspflege"}, inplace = True)
df_kp_pivoted.rename(columns={"Total":"LIK_Total"}, inplace = True)

# change column type to int
df_kp_pivoted['Jahr'] = df_kp_pivoted['Jahr'].astype('int')

df_kp_pivoted.head(5)


# In[950]:


# merge the LIK dataset to the final dataset
df_dg_kf_kp_merged = pd.merge(df_dg_kf_merged, df_kp_pivoted, 
                  on=['Jahr'], how='left')
df_dg_kf_kp_merged.head(50)


# In[951]:


# copy this one into a df_final
df_final = df_dg_kf_kp_merged.copy()
# last check of the structure
df_final.info()


# ### Imputation

# Da es nur eine geringe Menge von Inputdaten gibt, können Zeilen die Null-Werte enthalten nicht einfach entfernt werden (Eliminierungsverfahren). Das Ziel ist diese nochmal nach zu recherchieren, oder anhand der bestehenden Werte aus anderen Zeilen der Zeitreihe durch Imputation auffüllen zu können. Die folgende Grafik gibt einen Überblick über die fehlenden Werte und wie sie aufgefüllt werden sollen.
#   
# ![Alt-Text](./img/Datenmodellierung_Imputation.png "Imputation")

# #### Fehlende Werte LIK für 1981-1983

# Auf der Homepage vom Bundesamt für Statistik gibt es unzählige Tabellen wo der Landesindex der Konsumentenpreise in verschiedenen Aggregationsstufen und für unterschiedlichste Zeiträume abgebildet ist. Es gibt eine Tabelle, welche die volle Zeitreihe seit 1914 darstellt. Daraus kann man die fehlenden Werte für den totalen LIK und den Gesundheitspflege LIK 1981-1983 herauslesen. Die relevanten Datensets und einen Online-Rechner dazu findet man hier: 
# * [LIK, Totalindex auf allen Basen seit Einführung [LANGE REIHEN MULTIBASIS]](https://www.bfs.admin.ch/bfs/de/home/statistiken/preise/landesindex-konsumentenpreise/detailresultate.assetdetail.23772754.html "Quelle: Bundesamt für Statistik")
# * [LIK (September 1977=100), Detailresultate, Indexstand und Warenkorbstruktur 1977. [LIK77B77]](https://www.bfs.admin.ch/bfs/de/home/statistiken/preise/landesindex-konsumentenpreise/detailresultate.assetdetail.214279.html "Quelle: Bundesamt für Statistik")
# * [LIK-Online-Teuerungsrechner](https://lik-app.bfs.admin.ch/de/lik/rechner?periodType=Monatlich&start=11.2021&ende=11.2022&basis=AUTO "Quelle: Bundesamt für Statistik")
# 
# **LIK TOTAL**  
# 1981: 6.5  
# 1982: 5.7  
# 1983: 2.9  
# 
# **LIK Gesundheitspflege**  
# 1981: 6.1  
# 1982: 5.35  
# 1983: 2.6  
#   
# Diese Werte können nun manuell eingetragen/ergänzt werden im Datenset.

# In[952]:


# update values for LIK_Total
df_final.at[0,'LIK_Total'] = 6.5
df_final.at[1,'LIK_Total'] = 5.7
df_final.at[2,'LIK_Total'] = 2.9
# update values for LIK_Gesundheitspflege
df_final.at[0,'LIK_Gesundheitspflege'] = 6.1
df_final.at[1,'LIK_Gesundheitspflege'] = 5.35
df_final.at[2,'LIK_Gesundheitspflege'] = 2.6
# check the values
df_final.head(5)


# #### Fehlende Gesamtkosten für 2021

# Die Gesundheitskosten werden jeweils erst ca. 2 Jahre verzögert vom Bundesamt für Statistik herausgegeben. Die Zahlen für 2021 werden erst im April 2023 (also nach Abschluss dieser Arbeit) erwartet. Es gibt aber bereits erste Prognosen dafür: 
# * [Bericht Gesundheitsausgabenprognose Herbst 2021 der Konjunkturforschungsstelle der ETH Zürich (KOF)](https://ethz.ch/content/dam/ethz/special-interest/dual/kof-dam/documents/Medienmitteilungen/Gesundheitsausgaben/2021/Bericht_Gesundheitsausgabenprognose_Herbst_2021.pdf "Quelle: Konjunkturforschungsstelle der ETH Zürich (KOF)")
# 
# **Gesamtkosten Gesundheitswesen CH**  
# 2021: 91'037'000'000  
#   
# Dieser Wert kann nun manuell eingetragen/ergänzt werden im Datenset.

# In[953]:


# update value for LeistungenTotal 2021
df_final.at[40,'LeistungenTotal'] = 91037000000 
# check the values
df_final.tail(5)


# #### Fehlende Werte bzgl. Kosten 1981-1994 für Leistungen

# Auch nach längerer Recherche habe ich hier leider keine Aufteilungen mehr gefunden. Immerhin sind die Gesamtkosten für diese Jahre bekannt, wie auch die Verteilung der Altersgruppen. Anhand dieser Werte sollen die fehlenden Angaben statistisch errechnet werden.

# In[954]:


# setting the graph size globally
plt.rcParams['figure.figsize'] = (14, 10)

for col in df_final[["Ambulante Kurativbehandlung"
                     , "Gesundheitsgüter"
                     , "Langzeitpflege"
                     #, "LeistungenTotal"
                     , "Prävention"
                     , "Rehabilitation"
                     , "Stationäre Kurativbehandlung" 
                     , "Unterstützende Dienstleistungen"
                     , "Verwaltung"]].columns:
    plt.plot(df_final[col], linewidth=1, label=col)
    
plt.xlabel('Jahr', fontsize=16)
plt.ylabel('Kosten in CHF', fontsize=16)
plt.ticklabel_format(style='plain', useOffset=False, axis='y')
default_x_ticks = range(len(df_final["Jahr"]))
plt.xticks(default_x_ticks, df_final["Jahr"], fontsize=10, rotation = 60)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.set_cmap('Paired')
plt.show()


# Die grafische Darstellung zeigt, dass eine lineare Regression erfolgreich sein könnte. Bevor das versucht wird, sollen aber noch die prozentualen Anteile der Kosten der einzelnen Services an den Gesamtkosten untersucht werden. Da die Gesamtkosten für alle Jahre bekannt sind, wäre bei einem gleichmässigen Verlauf der prozentualen Anteile die Imputation sehr einfach erledigt.
#   
# Um die prozentualen Anteile zu berechnen, sollen nur die Kostenspalten und die Jahresspalte selektiert werden, danach wird jede der Kostenspalten verglichen mit den totalen Kosten und eine neue Spalte mit Präfix "perc_" erstellt, welche den berechneten Prozentwert beinhaltet.

# In[955]:


df_costs = df_final[["Jahr"
                     , "Ambulante Kurativbehandlung"
                     , "Gesundheitsgüter"
                     , "Langzeitpflege"
                     , "LeistungenTotal"
                     , "Prävention"
                     , "Rehabilitation"
                     , "Stationäre Kurativbehandlung" 
                     , "Unterstützende Dienstleistungen"
                     , "Verwaltung"]]

df_costs = df_costs.join(df_costs.iloc[:, 1:]                           .div(df_costs['LeistungenTotal'], axis=0)                           .mul(100)                           .add_prefix('perc_'))
df_costs.tail(10)


# In[956]:


# setting the graph size globally
plt.rcParams['figure.figsize'] = (14, 10)

for col in df_costs[["perc_Ambulante Kurativbehandlung"
                     , "perc_Gesundheitsgüter"
                     , "perc_Langzeitpflege"
                     #, "perc_LeistungenTotal"
                     , "perc_Prävention"
                     , "perc_Rehabilitation"
                     , "perc_Stationäre Kurativbehandlung" 
                     , "perc_Unterstützende Dienstleistungen"
                     , "perc_Verwaltung"]].columns:
    plt.plot(df_costs[col], linewidth=1, label=col)
    
plt.xlabel('Jahr', fontsize=16)
plt.ylabel('Prozentualer Anteil an Gesamtkosten', fontsize=16)
default_x_ticks = range(len(df_costs["Jahr"]))
plt.xticks(default_x_ticks, df_costs["Jahr"], fontsize=10, rotation = 60)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.set_cmap('Paired')
plt.show()


# Unter anderem aufgrund des Trends der Verlagerung von ambulanten zu stationären Eingriffen ist die Verteilung leider nicht sehr homogen.

# In[957]:


df_final.info()


# ##### Versuch 1: Imputation mit Linearer Regression

# In[958]:


# column configuration
columns_to_impute = ["Ambulante Kurativbehandlung"
                     , "Gesundheitsgüter"
                     , "Langzeitpflege"
                     , "Prävention"
                     , "Rehabilitation"
                     , "Stationäre Kurativbehandlung" 
                     , "Unterstützende Dienstleistungen"
                     , "Verwaltung"]

# filter to get only rows with 1 or more nan values
df_to_impute  = df_final[df_final.isna().any(axis=1)]
# filter to get only rows without nan values
df_train = df_final.dropna()

# loop through the configured columns needing an imputation
for x in range(len(columns_to_impute)):
    # create linear regression
    linreg = make_pipeline(MinMaxScaler(), LinearRegression())

    # create X and y
    X = df_train[['0-5','6-10','11-15','16-20','21-25','26-30','31-35','36-40','41-45','46-50',
                 '51-55','56-60','61-65','66-70','71-75','76-80','81-85','86-90','91-95','96+',
                 'LIK_Gesundheitspflege','LIK_Total','LeistungenTotal']]
    y = df_train[columns_to_impute[x]]

    # split into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # fit & predict 
    model = linreg.fit(X_train, y_train)
    predictions = linreg.predict(X_test)
    
    # print score
    print(f'Model score for {columns_to_impute[x]}:', model.score(X_test, y_test))
    # model evaluation
    print('-> mean_squared_error : ', mean_squared_error(y_test, predictions))
    print('-> mean_absolute_error : ', mean_absolute_error(y_test, predictions))


# Das lineare Regressionsmodell liefert trotz des kleinen Datensets zwar sehr gute Resultate für die Vorhersage der Kosten aller Leistungen (Bestimmtheitsmass > 0.9), ausser bei "Prävention". Dies ist erklärbar. Die Investitionen in die Prävention sind staatlich bestimmt, nicht direkt abhängig von der Anzahl Personen pro Altersgruppe und können über die Zeit auch stark schwanken. Leider sind aber die Werte für die mittlere absolute und mittlere quadratische Abweichung astronomisch hoch. Klar sind die Basis für die Vorhersagen Milliardenwerte, aber Abweichungen von 300 Mio. und grösser sind hier nicht akzeptabel.
#   
# Wenn die Abweichungen nicht so gross wären, dann könnte man an dieser Stelle - da wir die Gesamtkosten für die entsprechenden Jahre kennen - die Kostenwerte für alle Leistungen ausser "Prävention" mit dem linearen Modell vorhersagen. Die Kosten für die "Prävention" sind dann pro Jahr jeweils die Gesamtkosten abzüglich der Kosten aller anderen Services. So wäre auch die Konsistenz im Datensatz gegeben (Summer der Kosten aller Einzelleistungen entspricht den Gesamtkosten).

# In[959]:


# copy the dataframe
df_imputation = df_final.copy()

# filter to get only rows with 1 or more nan values
df_to_impute  = df_imputation[df_imputation.isna().any(axis=1)]
# filter to get only rows without nan values
df_train = df_imputation.dropna()

# column configuration
columns_to_impute = ["Ambulante Kurativbehandlung"
                     , "Gesundheitsgüter"
                     , "Langzeitpflege"
                     #, "Prävention" # skip this because it does not seem to be linear
                     , "Rehabilitation"
                     , "Stationäre Kurativbehandlung" 
                     , "Unterstützende Dienstleistungen"
                     , "Verwaltung"]

# loop through the configured columns needing an imputation
for x in range(len(columns_to_impute)):
    # create linear regression
    linreg = make_pipeline(MinMaxScaler(), LinearRegression())

    # create X and y
    X = df_train[['0-5','6-10','11-15','16-20','21-25','26-30','31-35','36-40','41-45','46-50',
                 '51-55','56-60','61-65','66-70','71-75','76-80','81-85','86-90','91-95','96+',
                 'LIK_Gesundheitspflege','LIK_Total','LeistungenTotal']]
    y = df_train[columns_to_impute[x]]

    # split into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # fit & predict 
    model = linreg.fit(X_train, y_train)
    predictions = linreg.predict(X_test)
    
    # print for better readability
    print(f'-------------------------------------------------------------------')
    print(f'-------------------{columns_to_impute[x].upper()}-----------------')
    print(f'-------------------------------------------------------------------')

    # print score
    print(f'- Model score for {columns_to_impute[x]}:', model.score(X_test, y_test))
    
    # get predicted values for the test data set
    print(f'--- Predictions for {columns_to_impute[x]} in the test dataset:',predictions)
    
    # get predicted values for the imputation
    predictions = linreg.predict(df_imputation[['0-5','6-10','11-15','16-20',
                  '21-25','26-30','31-35','36-40','41-45','46-50','51-55','56-60','61-65','66-70',
                  '71-75','76-80','81-85','86-90','91-95','96+','LIK_Gesundheitspflege','LIK_Total',
                  'LeistungenTotal']])
    print(f'--- Predictions for {columns_to_impute[x]}. The values to be imputed are:',predictions)
    
    # impute the values
    df_imputation[str(columns_to_impute[x]+'_pred')] = predictions


# Für Langzeitpflege, Rehabilitation und Stationäre Kurativbehandlung werden negative Werte vorhergesagt. Das kann bei einem linearen Modell natürlich immer passieren, ist aber in diesem spezifischen Fall wo es um Kosten geht nicht möglich.

# In[960]:


# setting the graph size globally
plt.rcParams['figure.figsize'] = (14, 10)

for col in df_imputation[[
                    #   "Ambulante Kurativbehandlung"
                    #, "Gesundheitsgüter"
                    #, "Langzeitpflege"
                    #, "Prävention"
                    #, "Rehabilitation"
                    #, "Stationäre Kurativbehandlung" 
                    #, "Unterstützende Dienstleistungen"
                    #, "Verwaltung"
                      "Ambulante Kurativbehandlung_pred"
                     , "Gesundheitsgüter_pred"
                     , "Langzeitpflege_pred"
                     , "Rehabilitation_pred"
                     , "Stationäre Kurativbehandlung_pred" 
                     , "Unterstützende Dienstleistungen_pred"
                     , "Verwaltung_pred"]].columns:
    plt.plot(df_imputation[col], linewidth=1, label=col)
    
plt.xlabel('Jahr', fontsize=16)
plt.ylabel('Kosten', fontsize=16)
plt.ticklabel_format(style='plain', useOffset=False, axis='y')
default_x_ticks = range(len(df_imputation["Jahr"]))
plt.xticks(default_x_ticks, df_imputation["Jahr"], fontsize=10, rotation = 60)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.set_cmap('Paired')
plt.show()


# Die grafische Darstellung zeigt, dass bei den Vorhersagen etwas nicht stimmen kann. Es soll also ein anderer Approach für die Imputation der fehlenden Werte gefunden werden.

# ##### Versuch 2: Imputation mit sklearn IterativeImputer

# In einem Rundlaufverfahren wird beim Iterative Imputer jeweils ein feature als Funktion der anderen Features modelliert. Die vorhergehenden Predictions werden in nachfolgenden Runden wieder verwendet. So wird insgesamt mehr Prozessierungszeit gebraucht, die Resultate können dadurch aber näher an den realen Werten sein.

# In[961]:


# copy the dataframe
df_imputation = df_final.copy()

# setting the random_state argument for reproducibility
imputer = IterativeImputer(random_state = 42, 
                           skip_complete = True,
                           sample_posterior = True, 
                           max_iter = 20, 
                           missing_values = np.nan, 
                           min_value = 0)
imputed = imputer.fit_transform(df_imputation)
df_imputed = pd.DataFrame(imputed, columns=df_imputation.columns)

# sum the predicted costs to get the total of the predictions per year
df_imputed['LeistungenTotalSum'] = df_imputed['Gesundheitsgüter'] + df_imputed['Langzeitpflege'] +                              df_imputed['Ambulante Kurativbehandlung'] + df_imputed['Rehabilitation'] +                              df_imputed['Stationäre Kurativbehandlung'] + df_imputed['Verwaltung'] +                              df_imputed['Unterstützende Dienstleistungen']+ df_imputed['Prävention']

# change column type to int
df_imputed['Jahr'] = df_imputed['Jahr'].astype('int')

# only print the columns that are interesting for a comparison
df_imputed[['Jahr','Ambulante Kurativbehandlung','Stationäre Kurativbehandlung','Langzeitpflege',
           'Rehabilitation','Gesundheitsgüter','Verwaltung','Prävention','Unterstützende Dienstleistungen',
           'LeistungenTotal','LeistungenTotalSum']]


# Interessanterweise stimmt die aus den resultierenden Werten der Imputation errechnete Summe sehr genau mit den effektiven Gesamtkosten überein. Bei näherer Betrachtung sieht man aber, dass z.B. für die "Verwaltung" plötzlich nur noch Kosten von 0 CHF vorhergesagt wurden. Dies liegt an der Konfiguration des minimalen Wertes von "0" im Modell (Option "min_value"). Wenn diese entfernt wird, werden aber wieder negative Werte vorhergesagt, was im Fall von Kosten wie bereits bei der linearen Regression beschrieben keinen Sinn macht.
#   
# Grafisch sieht das Ganze wie folgt aus:

# In[962]:


# setting the graph size globally
plt.rcParams['figure.figsize'] = (14, 10)

for col in df_imputed[[
                      "Ambulante Kurativbehandlung"
                     , "Gesundheitsgüter"
                     , "Langzeitpflege"
                     , "Rehabilitation"
                     , "Prävention"
                     , "Stationäre Kurativbehandlung" 
                     , "Unterstützende Dienstleistungen"
                     , "Verwaltung"]].columns:
    plt.plot(df_imputed[col], linewidth=1, label=col)
    
plt.xlabel('Jahr', fontsize=16)
plt.ylabel('Kosten', fontsize=16)
plt.ticklabel_format(style='plain', useOffset=False, axis='y')
default_x_ticks = range(len(df_imputed["Jahr"]))
plt.xticks(default_x_ticks, df_imputed["Jahr"], fontsize=10, rotation = 60)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.set_cmap('Paired')
plt.show()


# Die Vorhersagen sehen zwar grösstenteils ein bisschen realistischer aus als bei der linearen Regression, aber zufriedenstellend ist dies noch nicht.

# ##### Versuch 3: Imputation anhand prozentualer Anteile aus letztem verfügbarem Jahr und den Gesamtkosten

# Die fehlenden Werte sollen in diesem Versuch wie folgt errechnet werden:
# * 1981-1994: prozentuale Anteile aus dem Jahr 1995 nehmen und die Gesamtkosten von 1981-1994 jeweils pro Jahr entsprechend aufteilen
# * 2021: prozentuale Anteile aus dem Jahr 2020 nehmen und die Gesamtkosten von 2021 entsprechend aufteilen

# In[963]:


# copy the dataframe
df_imputation = df_final.copy()

# column configuration
columns_to_impute = ["Ambulante Kurativbehandlung"
                     , "Gesundheitsgüter"
                     , "Langzeitpflege"
                     , "Prävention"
                     , "Rehabilitation"
                     , "Stationäre Kurativbehandlung" 
                     , "Unterstützende Dienstleistungen"
                     , "Verwaltung"
                     , "LeistungenTotal"]

# get percentages from 1995
percentages = df_imputation.iloc[14, 21:30] / df_imputation.loc[14,'LeistungenTotal']

# loop through rows 0-13 (1981-1994)
for index, row in df_imputation.iloc[0:14, :].iterrows():
    # loop through the configured columns needing an imputation
    for x in range(len(columns_to_impute)):   
        df_imputation.at[index,columns_to_impute[x]] = row['LeistungenTotal'] * percentages[columns_to_impute[x]]

# get percentages from 2020
percentages = df_imputation.iloc[39, 21:30] / df_imputation.loc[39,'LeistungenTotal']

# loop through rows 40-? (2021)
for index, row in df_imputation.iloc[40:, :].iterrows():
    # loop through the configured columns needing an imputation
    for x in range(len(columns_to_impute)):   
        df_imputation.at[index,columns_to_impute[x]] = row['LeistungenTotal'] * percentages[columns_to_impute[x]]


# In[964]:


df_imputation


# Grafisch dargestellt sieht das so aus:

# In[965]:


# setting the graph size globally
plt.rcParams['figure.figsize'] = (14, 10)

for col in df_imputation[[
                      "Ambulante Kurativbehandlung"
                     , "Gesundheitsgüter"
                     , "Langzeitpflege"
                     , "Rehabilitation"
                     , "Prävention"
                     , "Stationäre Kurativbehandlung" 
                     , "Unterstützende Dienstleistungen"
                     , "Verwaltung"]].columns:
    plt.plot(df_imputation[col], linewidth=1, label=col)
    
plt.xlabel('Jahr', fontsize=16)
plt.ylabel('Kosten', fontsize=16)
plt.ticklabel_format(style='plain', useOffset=False, axis='y')
default_x_ticks = range(len(df_imputation["Jahr"]))
plt.xticks(default_x_ticks, df_imputation["Jahr"], fontsize=10, rotation = 60)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.set_cmap('Paired')
plt.show()


# Verglichen mit den anderen Imputations-Versuchen ist dies das realistischste Resultat. Daher soll damit weitergemacht werden.

# ### Ergänzung mit Daten der Bevölkerungsszenarien

# Datenset 3 liefert für die verschiedenen Bevölkerungsszenarien die Anzahl Menschen pro Altersgruppe, wofür dann die anderen Werte der Timeseries (Kosten / LIK) vorhergesagt werden sollen. Es soll für jedes Szenario ein Data Frame erstellt werden, wo die Bevölkerungsdaten bis 2070 bereits eingetragen sind, und die fehlenden bzw. zu vorhersagenden Werte mit nan aufgefüllt werden.

# In[966]:


# copy the dataframe
df_1981_2021 = df_imputation.copy()

# filter out 2019-2021
df_raw_be_pivoted_2022 = df_raw_be_pivoted                  .query('Jahr>=2022')                  .reset_index()

###### Scenario "Referenzszenario A-00-2020" ######

# filter scenarios for scenario 'Referenzszenario A-00-2020'
df_scenario_A_00_2020 = df_raw_be_pivoted_2022                  .query('`Szenario-Variante`=="Referenzszenario A-00-2020"')                  .reset_index()

# drop column 'Szenario-Variante'
df_scenario_A_00_2020.drop('Szenario-Variante', axis=1, inplace=True)

# drop the unnecessary index and level_0 column
df_scenario_A_00_2020.drop('index', axis=1, inplace=True)
df_scenario_A_00_2020.drop('level_0', axis=1, inplace=True)

# concat the two dataframes to have one ranging from 1981-2070
df_final_scenario_A_00 = pd.concat([df_1981_2021, df_scenario_A_00_2020], axis=0)

###### Scenario "hohes Szenario B-00-2020" ######

# filter scenarios for scenario 'hohes Szenario B-00-2020'
df_scenario_B_00_2020 = df_raw_be_pivoted_2022                  .query('`Szenario-Variante`=="\'hohes\' Szenario B-00-2020"')                  .reset_index()

# drop column 'Szenario-Variante'
df_scenario_B_00_2020.drop('Szenario-Variante', axis=1, inplace=True)

# drop the unnecessary index and level_0 column
df_scenario_B_00_2020.drop('index', axis=1, inplace=True)
df_scenario_B_00_2020.drop('level_0', axis=1, inplace=True)

# concat the two dataframes to have one ranging from 1981-2070
df_final_scenario_B_00 = pd.concat([df_1981_2021, df_scenario_B_00_2020], axis=0)

###### Scenario "tiefes Szenario C-00-2020" ######

# filter scenarios for scenario 'tiefes Szenario C-00-2020'
df_scenario_C_00_2020 = df_raw_be_pivoted_2022                  .query('`Szenario-Variante`=="\'tiefes\' Szenario C-00-2020"')                  .reset_index()

# drop column 'Szenario-Variante'
df_scenario_C_00_2020.drop('Szenario-Variante', axis=1, inplace=True)

# drop the unnecessary index and level_0 column
df_scenario_C_00_2020.drop('index', axis=1, inplace=True)
df_scenario_C_00_2020.drop('level_0', axis=1, inplace=True)

# concat the two dataframes to have one ranging from 1981-2070
df_final_scenario_C_00 = pd.concat([df_1981_2021, df_scenario_C_00_2020], axis=0)

###### Scenario "verstärkte Alterung D-00-2020" ######

# filter scenarios for scenario 'verstärkte Alterung D-00-2020'
df_scenario_D_00_2020 = df_raw_be_pivoted_2022                  .query('`Szenario-Variante`=="Szenario D-00-2020 \'verstärkte Alterung\'"')                  .reset_index()

# drop column 'Szenario-Variante'
df_scenario_D_00_2020.drop('Szenario-Variante', axis=1, inplace=True)

# drop the unnecessary index and level_0 column
df_scenario_D_00_2020.drop('index', axis=1, inplace=True)
df_scenario_D_00_2020.drop('level_0', axis=1, inplace=True)

# concat the two dataframes to have one ranging from 1981-2070
df_final_scenario_D_00 = pd.concat([df_1981_2021, df_scenario_D_00_2020], axis=0)

###### Scenario "höhere Lebenserwartung bei der Geburt A-03-2020" ######

# filter scenarios for scenario 'höhere Lebenserwartung bei der Geburt A-03-2020'
df_scenario_A_03_2020 = df_raw_be_pivoted_2022                  .query('`Szenario-Variante`=="Variante A-03-2020 \'höhere Lebenserwartung bei der Geburt\'"')                  .reset_index()

# drop column 'Szenario-Variante'
df_scenario_A_03_2020.drop('Szenario-Variante', axis=1, inplace=True)

# drop the unnecessary index and level_0 column
df_scenario_A_03_2020.drop('index', axis=1, inplace=True)
df_scenario_A_03_2020.drop('level_0', axis=1, inplace=True)

# concat the two dataframes to have one ranging from 1981-2070
df_final_scenario_A_03 = pd.concat([df_1981_2021, df_scenario_A_03_2020], axis=0)


# ### Zusammenfassung

# Schlussendlich besteht jetzt also pro Szenario ein Data Frame mit 41 Zeilen (Jahr 1981-2021), der Anzahl Menschen pro Altersgruppen (5-Jahres-Gruppen), den Kosten pro Service und Jahr, sowie dem Landesindex der Konsumentenpreise (total und nur "Gesundheitspflege") pro Jahr. 
#   
# Die nachfolgende Grafik veranschaulicht die obigen Ausführungen.

# ![Alt-Text](./img/Datenmodellierung_Zielmodell.png "Zielmodell & Plan für die Analyse")

# Die verschiedenen Data Frames kann man auch in einem Diagramm darstellen um die Verläufe der einzelnen Szenarien zu zeigen:

# In[967]:


# get sums for each year (population)
df_1981_2021['Total'] = df_1981_2021.iloc[:, 1:21].sum(axis=1)
df_final_scenario_A_00['Total'] = df_final_scenario_A_00.iloc[:, 1:21].sum(axis=1)
df_final_scenario_B_00['Total'] = df_final_scenario_B_00.iloc[:, 1:21].sum(axis=1)
df_final_scenario_C_00['Total'] = df_final_scenario_C_00.iloc[:, 1:21].sum(axis=1)
df_final_scenario_D_00['Total'] = df_final_scenario_D_00.iloc[:, 1:21].sum(axis=1)
df_final_scenario_A_03['Total'] = df_final_scenario_A_03.iloc[:, 1:21].sum(axis=1)

# plot in matplotlib
plt.plot(df_1981_2021['Jahr'], 
         df_1981_2021['Total'], 
         'black', 
         df_final_scenario_A_00.query('Jahr>=2021')['Jahr'], 
         df_final_scenario_A_00.query('Jahr>=2021')['Total'], 
         'blue', 
         df_final_scenario_B_00.query('Jahr>=2021')['Jahr'], 
         df_final_scenario_B_00.query('Jahr>=2021')['Total'], 
         'red', 
         df_final_scenario_C_00.query('Jahr>=2021')['Jahr'], 
         df_final_scenario_C_00.query('Jahr>=2021')['Total'], 
         'green', 
         df_final_scenario_D_00.query('Jahr>=2021')['Jahr'], 
         df_final_scenario_D_00.query('Jahr>=2021')['Total'], 
         'orange', 
         df_final_scenario_A_03.query('Jahr>=2021')['Jahr'], 
         df_final_scenario_A_03.query('Jahr>=2021')['Total'], 
         'fuchsia'
        )

# add a legend for the plot
plt.legend(["Base Data",
            "Referenzszenario A-00-2020",
            "'hohes' Szenario B-00-2020",
            "'tiefes' Szenario C-00-2020",
            "'verstärkte Alterung' D-00-2020",
            "'höhere Lebenserwartung bei der Geburt' A-03-2020"])

# prevent scientific notation
plt.ticklabel_format(style='plain', useOffset=False, axis='y')

# show the plot
plt.show() 


# ## Modellentwicklung & Prognose

# Es gibt eine ganze Reihe von Modellen, die auf Timeseries Daten angewendet werden können. Die Hauptschwierigkeit dieser Arbeit ist die für das Trainieren und Testen der Modelle sehr limitierte Menge an Daten. In den folgenden Abschnitten sollen einige Timeseries Modelle trainiert, getestet und kritisch bewertet werden.
#   
# Zur Übersicht nochmal alle vorbereiteten Datensets:

# In[968]:


# drop the previously created columns for the total population counts
df_final_scenario_A_00.drop('Total', axis=1, inplace=True)
df_final_scenario_B_00.drop('Total', axis=1, inplace=True)
df_final_scenario_C_00.drop('Total', axis=1, inplace=True)
df_final_scenario_D_00.drop('Total', axis=1, inplace=True)
df_final_scenario_A_03.drop('Total', axis=1, inplace=True)


# Wichtig ist bei allen Modellen aber, dass nicht die train_test_split Methode verwendet wird um die Trainings-/Testdatensätze zu erstellen. Da es sich um Timeseries-Daten handelt, können die Daten nicht einfach zufällig in die einzelnen Gruppen eingeteilt werden. Man muss jeweils die ersten z.B. 80% Zeilen von den letzten 20% Zeilen trennen, damit man die kontinuierliche(n) Zeitreihe(n) nicht zerstört.

# ### Recursive multi-step forecasting with exogenous variables

# Dieses Modell ist Teil vom skforecast Modul welches von Joaquin Amat Rodrigo entwickelt wurde. Das Modul ist speziell für das Forecasting von Daten in Timeseries Szenarien da. Da in den Daten dieser Arbeit die Vorhersagen auch anhand von bereits errechneten Zukunftsszenarien gemacht werden sollen, wurde vom Modul die Methode "recursive multi-step forecasting with exogenous variables" ausgewählt. Die exogenen Variablen sind hier die Grössen der einzelnen Altersgruppen.

# Quellen: 
# * https://joaquinamatrodrigo.github.io/skforecast/0.3/guides/autoregresive-forecaster-exogenous.html

# Einzelne Code-Abschnitte wurden aus der offiziellen Dokumentation des Moduls und weiterführenden Quellen kopiert und wo nötig für die Zwecke dieser Arbeit adaptiert.

# In[969]:


# copy the input dataset
scenario_A_00 = df_final_scenario_A_00.copy()


# In[970]:


# reset index
scenario_A_00 = scenario_A_00.reset_index()


# In[971]:


# split input data to create train and test dataset
df_scenario_A_00_train, df_scenario_A_00_test = scenario_A_00[1:36], scenario_A_00[36:]


# In[972]:


# create and fit forecaster with 10 lags (10 last observations used to predict the next one)
forecaster = ForecasterAutoreg(
                    regressor = Ridge(),
                    lags      = 10
                )
# fit the model with the train dataset
forecaster.fit(
    y    = df_scenario_A_00_train['LeistungenTotal'],
    exog = df_scenario_A_00_train[['0-5','6-10','11-15','16-20','21-25','26-30','31-35',
                                   '36-40','41-45','46-50','51-55','56-60','61-65','66-70',
                                   '71-75','76-80','81-85','86-90','91-95','96+']]
)
# check the model details
forecaster


# In[973]:


# predict next 54 values in timeseries
steps = 54
predictions = forecaster.predict(
                steps = steps,
                exog = df_scenario_A_00_test[['0-5','6-10','11-15','16-20','21-25','26-30','31-35',
                                   '36-40','41-45','46-50','51-55','56-60','61-65','66-70',
                                   '71-75','76-80','81-85','86-90','91-95','96+']]
               )
# add datetime index to predictions
predictions = pd.Series(data=predictions, index=df_scenario_A_00_test.index)
predictions.head(5)


# In[974]:


# plot the predictions
fig, ax=plt.subplots(figsize=(9, 4))
df_scenario_A_00_train['LeistungenTotal'].plot(ax=ax, label='train')
df_scenario_A_00_test['LeistungenTotal'].plot(ax=ax, label='test')
predictions.plot(ax=ax, label='predictions')
ax.legend();


# In der Grafik ist ersichtlich, dass die Vorhersage und die realen Werte aus dem Testdatenset sehr gut übereinstimmen. Die Kostenentwicklung ist auch in etwa so, wie man sie hätte erwarten können. Dies kann noch mathematisch geprüft werden:

# In[978]:


mae = mean_absolute_error(df_scenario_A_00_test
                    .query('Jahr>=2017')
                    .head(5)
                    .LeistungenTotal, 
                    predictions.head(5))
mean_cost = df_scenario_A_00_test.query('Jahr>=2017').head(5).LeistungenTotal.mean()
error = mae/mean_cost

print(f'Mean absolute error of predictions is: {mae}')
print(f'Mean total costs of actual values is: {mean_cost}')
print(f'This gives an error percentage of {error*100} percent.')


# Die vorhergesagten Werte für dieses Szenario sollen in einem neuen Dataframe persistiert werden.

# In[979]:


df_all_scenarios_pred = df_scenario_A_00_test.query('Jahr>=2022')[['Jahr']]
df_all_scenarios_pred=df_all_scenarios_pred.merge(predictions.rename('A_00_Predicted'),
             left_index=True, right_index=True)
df_all_scenarios_pred.head(5)


# Nun soll das Modell erweitert werden für die Kostenvorhersagen der restlichen Bevölkerungsszenarien.

# #### Erweitert um andere Szenarien

# In[980]:


# copy input data
scenario_D_00 = df_final_scenario_D_00.copy()
# reset index
scenario_D_00 = scenario_D_00.reset_index()
# split input data to create train and test dataset
df_scenario_D_00_train, df_scenario_D_00_test = scenario_D_00[1:36], scenario_D_00[36:]

# predict next 54 steps in timeseries
steps = 54
predictions = forecaster.predict(
                steps = steps,
                exog = df_scenario_D_00_test[['0-5','6-10','11-15','16-20','21-25','26-30','31-35',
                                   '36-40','41-45','46-50','51-55','56-60','61-65','66-70',
                                   '71-75','76-80','81-85','86-90','91-95','96+']]
               )
# add datetime index to predictions
predictions = pd.Series(data=predictions, index=df_scenario_D_00_test.index)

# add results to all_scenarios dataset
df_all_scenarios_pred=df_all_scenarios_pred.merge(predictions.rename('D_00_Predicted'),
             left_index=True, right_index=True)

##################################################

# copy input data
scenario_B_00 = df_final_scenario_B_00.copy()
# reset index
scenario_B_00 = scenario_B_00.reset_index()
# split input data to create train and test dataset
df_scenario_B_00_train, df_scenario_B_00_test = scenario_B_00[1:36], scenario_B_00[36:]

# predict next 54 steps in timeseries
steps = 54
predictions = forecaster.predict(
                steps = steps,
                exog = df_scenario_B_00_test[['0-5','6-10','11-15','16-20','21-25','26-30','31-35',
                                   '36-40','41-45','46-50','51-55','56-60','61-65','66-70',
                                   '71-75','76-80','81-85','86-90','91-95','96+']]
               )
# add datetime index to predictions
predictions = pd.Series(data=predictions, index=df_scenario_B_00_test.index)

# add results to all_scenarios dataset
df_all_scenarios_pred=df_all_scenarios_pred.merge(predictions.rename('B_00_Predicted'),
             left_index=True, right_index=True)

##################################################

# copy input data
scenario_C_00 = df_final_scenario_C_00.copy()
# reset index
scenario_C_00 = scenario_C_00.reset_index()
# split input data to create train and test dataset
df_scenario_C_00_train, df_scenario_C_00_test = scenario_C_00[1:36], scenario_C_00[36:]

# predict next 54 steps in timeseries
steps = 54
predictions = forecaster.predict(
                steps = steps,
                exog = df_scenario_C_00_test[['0-5','6-10','11-15','16-20','21-25','26-30','31-35',
                                   '36-40','41-45','46-50','51-55','56-60','61-65','66-70',
                                   '71-75','76-80','81-85','86-90','91-95','96+']]
               )
# add datetime index to predictions
predictions = pd.Series(data=predictions, index=df_scenario_C_00_test.index)

# add results to all_scenarios dataset
df_all_scenarios_pred=df_all_scenarios_pred.merge(predictions.rename('C_00_Predicted'),
             left_index=True, right_index=True)

##################################################

# copy input data
scenario_A_03 = df_final_scenario_A_03.copy()
# reset index
scenario_A_03 = scenario_A_03.reset_index()
# split input data to create train and test dataset
df_scenario_A_03_train, df_scenario_A_03_test = scenario_A_03[1:36], scenario_A_03[36:]

# predict next 54 steps in timeseries
steps = 54
predictions = forecaster.predict(
                steps = steps,
                exog = df_scenario_A_03_test[['0-5','6-10','11-15','16-20','21-25','26-30','31-35',
                                   '36-40','41-45','46-50','51-55','56-60','61-65','66-70',
                                   '71-75','76-80','81-85','86-90','91-95','96+']]
               )
# add datetime index to predictions
predictions = pd.Series(data=predictions, index=df_scenario_A_03_test.index)

# add results to all_scenarios dataset
df_all_scenarios_pred=df_all_scenarios_pred.merge(predictions.rename('A_03_Predicted'),
             left_index=True, right_index=True)

# check the all_scenarios dataset after all predictions
df_all_scenarios_pred.head(5)


# Nun soll das Datenset mit den vorhergesagten Werten für alle Szenarien grafisch dargestellt werden.

# In[981]:


# drop all nan values
scenario_A_00 = df_final_scenario_A_00.dropna()

# plot in matplotlib
plt.plot(scenario_A_00['Jahr'], 
         scenario_A_00['LeistungenTotal'], 
         'black',
         df_all_scenarios_pred['Jahr'], 
         df_all_scenarios_pred['A_00_Predicted'], 
         'blue', 
         df_all_scenarios_pred['Jahr'], 
         df_all_scenarios_pred['B_00_Predicted'], 
         'red',
         df_all_scenarios_pred['Jahr'], 
         df_all_scenarios_pred['C_00_Predicted'], 
         'green',
         df_all_scenarios_pred['Jahr'], 
         df_all_scenarios_pred['D_00_Predicted'], 
         'orange', 
         df_all_scenarios_pred['Jahr'], 
         df_all_scenarios_pred['A_03_Predicted'],
         'fuchsia'
        )

# add a legend for the plot
plt.legend(["Base Data",
            "Referenzszenario A-00-2020",
            "'hohes' Szenario B-00-2020",
            "'tiefes' Szenario C-00-2020",
            "'verstärkte Alterung' D-00-2020",
            "'höhere Lebenserwartung bei der Geburt' A-03-2020"])

# prevent scientific notation
plt.ticklabel_format(style='plain', useOffset=False, axis='y')

# show the plot
plt.show() 


# Erstaunlich wirkt hier vor Allem die Entwicklung im Szenario "verstärkte Alterung". Intuitiv wäre man hier davon ausgegangen, dass dadurch die Gesundheitskosten mittelfristig stark ansteigen würden. Dies scheint aber nicht der Fall zu sein. Die restlichen Vorhersagen sehen so stimmig aus. Das Modell macht einen sehr guten Eindruck.

# ### Lineare Regression

# Da man es im Machine Learning Bereich immer erstmal mit den einfacheren Modellen versuchen soll, wird an dieser Stelle auch eine Lineare Regression gemacht.

# Quellen: 
# * https://www.linkedin.com/pulse/using-python-create-multivariate-linear-regression-model-sibanda

# In einem ersten Schritt muss das Datenset in einen Trainings- und einen Testdatensatz aufgeteilt werden.

# In[982]:


# drop nan rows
scenario_A_00 = df_final_scenario_A_00.dropna()


# In[983]:


# set the year column values as index
scenario_A_00_i = scenario_A_00.set_index('Jahr')


# In[984]:


# split input data to create train and test dataset
df_scenario_A_00_train, df_scenario_A_00_test = scenario_A_00_i[1:36], scenario_A_00_i[36:]


# Das Modell kann nun mit den Daten trainiert und getestet werden.

# In[985]:


# independent and dependent variables
features= ['0-5','6-10','11-15','16-20','21-25','26-30','31-35','36-40',
           '41-45','46-50','51-55','56-60','61-65','66-70','71-75','76-80',
           '81-85','86-90','91-95','96+','LIK_Total','LIK_Gesundheitspflege']
target = 'LeistungenTotal'
# define model
model = LinearRegression()
# training process
model.fit(df_scenario_A_00_train[features], df_scenario_A_00_train[target])
model.fit(df_scenario_A_00_test[features], df_scenario_A_00_test[target])


# Für die Evaluierung des Modelles soll der mittlere absolute Fehler berechnet werden.

# In[986]:


# mean absolute value for training data
data = df_scenario_A_00_train[target]
predict =  model.predict(df_scenario_A_00_train[features])
training_error = mean_absolute_error(data, predict)
# mean absolute value for test data
test_data = df_scenario_A_00_test[target]
predict_test = model.predict(df_scenario_A_00_test[features])
test_data_error = mean_absolute_error(test_data, predict_test)


# In[987]:


# on training data
true_value = df_scenario_A_00_train[target]
predicted_val =  model.predict(df_scenario_A_00_train[features])
accuracy = r2_score(true_value, predicted_val)
# on test data
true_value2 = df_scenario_A_00_test[target]
predicted_val2 =  model.predict(df_scenario_A_00_test[features])
accuracy2 = r2_score(true_value2, predicted_val2)


# Diese können mit den folgenden zwei Aufrufen ausgegeben werden:

# In[988]:


print('This model accounts for {}% of the training data with mean data error of {}'.format(round(accuracy2*100,2), round(training_error,2)))
print('This model accounts for {}% of the testing data with mean data error of {}'.format(round(accuracy*100,2), round(test_data_error,2)))


# Wie man unschwer erkennen kann, performt das lineare Modell gar nicht gut. Auch mit längerem Asuprobieren von anderen Zusammenstellungen der Trainings-/Testdaten, kommt kein besseres Resultat zustande. Der Ansatz mit der linearen Regression wird deswegen verworfen, hier aber als "gescheiterter Versuch" in der Arbeit behalten.

# ### Vector Auto Regression (VAR)

# Das Vector Auto Regression (VAR) Modell ist insofern speziell, als dass die Vorhersagen für eine Variable anhand der vorhergehenden Werte der Variable selbst, aber auch der vorhergehenden Werte der anderen Variablen bestimmt wird. Das Modell funktioniert also besonders gut, wenn zwei oder mehr verschiedene Zeitreihen sich gegenseitig beeinflussen.
# 
# Sehr wichtig ist jedoch, dass die einzelnen Variablen stationär sind. Das heisst, dass die Verteilung der Werte nicht zeitabhängig ist (also nicht wie z.B. bei Allergiedaten, wo jeweils im Frühling die Werte naturgemäss höher sind als sonst im Jahr). 

# Quellen: 
# * https://www.analyticsvidhya.com/blog/2018/09/multivariate-time-series-guide-forecasting-modeling-python-codes/
# * https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/

# Einzelne Code-Abschnitte wurden aus der offiziellen Dokumentation des Moduls und weiterführenden Quellen kopiert und wo nötig für die Zwecke dieser Arbeit adaptiert.

# In[989]:


# drop nan rows
scenario_A_00 = df_final_scenario_A_00.dropna()


# In[990]:


# set year column values as index
scenario_A_00_i = scenario_A_00.set_index('Jahr')
# drop unnecessary columns
scenario_A_00_i.drop('LIK_Gesundheitspflege', axis=1, inplace=True)
scenario_A_00_i.drop('LIK_Total', axis=1, inplace=True)


# Mit der bereits etablierten Methodik soll wieder der Trainings- und Testdatensatz erstellt werden.

# In[991]:


df_scenario_A_00_train, df_scenario_A_00_test = scenario_A_00_i[1:33], scenario_A_00_i[33:] 


# Bevor mit der Modellentwicklung definitiv gestartet wird, sollen nochmal alle Zeitreihen grafisch dargestellt werden:

# In[992]:


# Plot
fig, axes = plt.subplots(nrows=7, ncols=4, dpi=120, figsize=(10,10))
for i, ax in enumerate(axes.flatten()):
    data = scenario_A_00_i[scenario_A_00_i.columns[i]]
    ax.plot(data, color='red', linewidth=1)
    # Decorations
    ax.set_title(scenario_A_00_i.columns[i], fontsize=6)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout();


# Mit einem Chi-Quadrat-Test soll auch noch statistisch geprüft werden, ob es Abhängigkeiten zwischen den einzelnen Variablen gibt.

# In[993]:


maxlag=10
test = 'ssr_chi2test'
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

grangers_causation_matrix(scenario_A_00_i, variables = scenario_A_00_i.columns)  


# Anhand der p-Werte für die einzelnen Variablenkombinationen kann evaluiert werden, ob die Nullhypothese ("es gibt keinen Zusammenhang zwischen den beiden Variablen") verworfen werden kann. Die Variablen mit Wert kleiner als 0.05 (5%) beeinflussen sich also. Wie man in der Matrix sieht ist das bei praktisch allen Kombinationen der Fall.

# Eine weitere wichtige Prüfung ist, dass keine der Variablen nicht-stationär ist. Dies kann mit dem Adfuller-Test geprüft werden:

# In[994]:


def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")


# Die Resultate für jede Spalte sollen ausgegeben werden:

# In[995]:


# ADF Test on each column
for name, column in scenario_A_00_i.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')


# Leider gibt es etliche Spalten, die nicht stationär sind. Es gibt die Möglichkeit die Differenz zwischen den einzelnen aufeinanderfolgenden Observationen zu berechnen, dadurch kann versucht werden, die nicht-stationären Spalten stationär zu machen. Dies kann auch mehrfach angewandt werden.

# In[996]:


# 1st difference
df_differenced = scenario_A_00_i.diff().dropna()


# In[997]:


# ADF Test on each column
for name, column in df_differenced.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')


# Die erste Anwendung war leider noch nicht erfolgreich. Also ein zweiter Vorgang:

# In[998]:


# 2nd difference
df_differenced = df_differenced.diff().dropna()


# In[999]:


# ADF Test on each column
for name, column in df_differenced.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')


# Auch der zweite Differenzier-Vorgang war nicht erfolgreich. Da das Modell nicht auf nicht-stationäre Daten angewendet werden kann, wird der Vector Autoregression Ansatz deswegen verworfen, hier aber als "gescheiterter Versuch" in der Arbeit behalten.

# ### Long-/Short-Term Memory (LSTM)

# Das Long-/Short-Term Memory (LSTM) Modell ist eine spezielle Form eines rückgekoppelten neuronalen Netzwerks. Es wird häufig dazu genutzt Muster in datensätzen zu erkennen und ist besonders beliebt bei Timeseries-Daten (z.B. Aktienkurse).

# Quellen: 
# * https://www.kaggle.com/code/sasakitetsuya/multivariate-time-series-forecasting-with-lstms
# * https://www.analyticsvidhya.com/blog/2020/10/multivariate-multi-step-time-series-forecasting-using-stacked-lstm-sequence-to-sequence-autoencoder-in-tensorflow-2-0-keras/

# Einzelne Code-Abschnitte wurden aus der offiziellen Dokumentation des Moduls und weiterführenden Quellen kopiert und wo nötig für die Zwecke dieser Arbeit adaptiert.

# In[1000]:


# load dataset
values = df_final_scenario_A_00.values
# specify columns to plot
groups = [21, 22, 23, 24, 25, 26, 27, 28, 29]
i = 1
# plot each column
plt.figure(figsize=(15,15))
for group in groups:
    plt.subplot(len(groups), 1, i)
    plt.plot(values[:, group])
    plt.title(df_final_scenario_A_00.columns[group], y=0.5, loc='right')
    i += 1
plt.show()


# Vorbereitung der Inputdaten:

# In[1001]:


# drop nan rows
scenario_A_00 = df_final_scenario_A_00.dropna()


# In[1002]:


# create index with year column values
scenario_A_00 = scenario_A_00.set_index('Jahr')


# In[1003]:


# split input data into train and test dataset
df_scenario_A_00_train, df_scenario_A_00_test = scenario_A_00[1:27], scenario_A_00[27:] 


# Normalisierung der Daten auf Werte zwischen 0 und 1:

# In[1004]:


# disable warnings about chained_assignments
pd.set_option('mode.chained_assignment', None)
scalers={}

# scale the train data with MinMaxScaler
train = df_scenario_A_00_train
for i in df_scenario_A_00_train.columns:
    scaler = MinMaxScaler(feature_range=(0,1))
    s_s = scaler.fit_transform(train[i].values.reshape(-1,1))
    s_s=np.reshape(s_s,len(s_s))
    scalers['scaler_'+ i] = scaler
    train[i]=s_s

# scale the test data with MinMaxScaler
test = df_scenario_A_00_test
for i in df_scenario_A_00_test.columns:
    scaler = scalers['scaler_'+i]
    s_s = scaler.transform(test[i].values.reshape(-1,1))
    s_s=np.reshape(s_s,len(s_s))
    scalers['scaler_'+i] = scaler
    test[i]=s_s


# Um die Daten in Series aufzuteilen, wo immer die letzten n Werte zusammen mit den nächsten (zukünftigen) n Werten aufgeführt sind, kann folgende Funktion verwendet werden:

# In[1005]:


# function to split the data into series with last n past records and future n records
def split_series(series, n_past, n_future):
    X, y = list(), list()
    for window_start in range(len(series)):
        past_end = window_start + n_past
        future_end = past_end + n_future
        if future_end > len(series):
            break
        past, future = series[window_start:past_end, :], series[past_end:future_end, :]
        X.append(past)
        y.append(future)
    return np.array(X), np.array(y)


# Anhand der letzten 10 Werte/Zeitschritten und dazugehörigen 31 Features sollen jeweils die 3 nächsten Werte vorhergesagt werden:

# In[1006]:


# configurations for number of past/future records and the number of features
n_past = 10
n_future = 3 
n_features = 31


# Aufteilen des Trainings-/Testdatensatzes in X und y "Vektoren":

# In[1007]:


# split data into X and y vectors for train dataset
X_train, y_train = split_series(df_scenario_A_00_train.values,n_past, n_future)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],n_features))
y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], n_features))
# split data into X and y vectors for test dataset 
X_test, y_test = split_series(df_scenario_A_00_test.values,n_past, n_future)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],n_features))
y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], n_features))


# Nun soll ein Modell definiert werden, mit zwei Encoder und zwei Decoder Layer:

# In[1008]:


# E1D1
# n_features ==> no of features at each timestep in the data.
#
encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
encoder_l1 = tf.keras.layers.LSTM(100,return_sequences = True, return_state=True)
encoder_outputs1 = encoder_l1(encoder_inputs)
encoder_states1 = encoder_outputs1[1:]
encoder_l2 = tf.keras.layers.LSTM(100, return_state=True)
encoder_outputs2 = encoder_l2(encoder_outputs1[0])
encoder_states2 = encoder_outputs2[1:]
#
decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs2[0])
#
decoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_inputs,initial_state = encoder_states1)
decoder_l2 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_l1,initial_state = encoder_states2)
decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(decoder_l2)
#
model_e1d1 = tf.keras.models.Model(encoder_inputs,decoder_outputs2)
#
model_e1d1.summary()


# Mit den folgenden Befehlen kann das Modell kompiliert und trainiert werden:

# In[1009]:


reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)
model_e1d1.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber())
history_e1d1=model_e1d1.fit(X_train,y_train,epochs=100,validation_data=(X_test,y_test),batch_size=32,verbose=0,callbacks=[reduce_lr])


# Vorhersagen für den Testdatensatz:

# In[1010]:


pred_e1d1=model_e1d1.predict(X_test)


# Um die Daten besser verstehen zu können, müssen sie wieder zurücktransformiert werden. Es soll also die inverse_transform Funktion aufgerufen werden, um die Normalisierung mit dem MinMaxScaler oben wieder zurückzurechnen. Wichtig ist, dass beide Vorgänge stochastisch funktionieren und man niemals wieder exakt dieselben Werte bekommen wird.

# In[1011]:


for index,i in enumerate(df_scenario_A_00_train.columns):
    scaler = scalers['scaler_'+i]
    pred_e1d1[:,:,index]=scaler.inverse_transform(pred_e1d1[:,:,index])
    y_train[:,:,index]=scaler.inverse_transform(y_train[:,:,index])
    y_test[:,:,index]=scaler.inverse_transform(y_test[:,:,index])


# Nun kann für jede Spalte der mittlere absolute Fehler für die drei Vorhersagen (Jahr 1, Jahr 2 und Jahr 3) ausgegeben werden.

# In[1012]:


for index,i in enumerate(df_scenario_A_00_train.columns):
    print(i)
    for j in range(1,4):
        print("Jahr ",j,":")
        print("MAE-E1D1 : ",mean_absolute_error(y_test[:,j-1,index],pred_e1d1[:,j-1,index]))
    print()


# Die Resultate sind sehr ernüchternd. Auch mit anderen Parametern oder einem Anpassen des Modelles konnte leider kein besseres Resultat erzielt werden. Dieser LSTM Ansatz wird deswegen auch verworfen, hier aber als "gescheiterter Versuch" in der Arbeit behalten.

# ### Multivariate time series forecasting with multiple lag inputs (LSTM)

# Nach längerer Recherche ergab sich noch folgender Versuch. Es ist auch ein LSTM Ansatz, aber das Vorgehen ist bisschen anders als im vorangehenden Beispiel. Die Daten werden für das LSTM Modell anders aufgeteilt.

# Quellen: 
# * https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

# Einzelne Code-Abschnitte wurden aus der offiziellen Dokumentation des Moduls und weiterführenden Quellen kopiert und wo nötig für die Zwecke dieser Arbeit adaptiert.

# Vorbereitung der Daten:

# In[1013]:


# copy the input dataset
scenario_A_00 = df_final_scenario_A_00.copy()


# In[1014]:


# set the index with year column values
scenario_A_00 = scenario_A_00.set_index('Jahr')


# In[1015]:


# reduce the dataset to the age group values and the total costs
scenario_A_00 = scenario_A_00[['0-5','6-10','11-15','16-20',
                  '21-25','26-30','31-35','36-40','41-45','46-50','51-55','56-60','61-65','66-70',
                  '71-75','76-80','81-85','86-90','91-95','96+','LeistungenTotal']]


# Hier kommt eine neue Methode für die Aufteilung der Daten. Die einzelnen Zeitschritte und dazugehörige Werte werden hier in einzelne Spalten geschrieben (t-10,t-9, etc.).

# In[1016]:


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
     # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# Nun soll die Methode ausprobiert werden mit den Daten, die auch hier zuerst noch mit dem MinMaxScaler auf Werte zwischen 0 und 1 normalisiert werden.

# In[1017]:


# load values
values = scenario_A_00.values

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# specify the number of lag years
n_years = 5
n_features = 20
# frame as supervised learning
reframed = series_to_supervised(scaled, n_years, 1)
print(reframed.shape)


# Die transformierten Daten können nun in einen Trainings-/Testdatensatz aufgeteilt werden.

# In[1018]:


# split into train and test sets
values = reframed.values
n_train_years = 30
train = values[:n_train_years, :]
test = values[n_train_years:, :]
# split into input and outputs
n_obs = n_years * n_features
train_X, train_y = train[:, :n_obs], train[:, -1]
test_X, test_y = test[:, :n_obs], test[:, -1]
print(train_X.shape, len(train_X), train_y.shape)
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_years, n_features))
test_X = test_X.reshape((test_X.shape[0], n_years, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# Das Modell besteht aus einem LSTM Layer und zwei Dense Layer mit Aktivierungsfunktionen.

# In[1019]:


# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(10, activation="relu"))
model.add(Dense(1, activation="linear"))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=150, validation_data=(test_X, test_y), verbose=2, shuffle=False)


# Wenn man die Kostenfunktionen darstellt, ergibt sich folgendes Bild:

# In[1020]:


# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


# Nun können die vorhersagen für das Testdatenset gemacht werden:

# In[1021]:


# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_years*n_features))


# Damit die Daten besser interpretiert werden können, wird auch hier die Normalisierung wieder rückgängig gemacht:

# In[1022]:


# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, 1:scaled.shape[1]]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,-1]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 1:scaled.shape[1]]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,-1]


# Als Mass für die Prognosegüte wird der Root Mean Square Error (RMSE) ausgegeben. Je grösser der RMSE ist, desto grösser ist die Differenz zwischen den vorhergesagten und den effektiv beobachteten Werten.

# In[1023]:


# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


# ### Auto-Regressive Integrated Moving Average (ARIMA)

# Das Modell Auto-Regressive Integrated Moving Average wird zur statistischen Analyse von Timeseries Daten verwendet.

# Quellen: 
# * https://www.kaggle.com/code/sajikim/time-series-forecasting-methods-example-python/notebook
# * https://www.section.io/engineering-education/multivariate-time-series-using-auto-arima/

# Einzelne Code-Abschnitte wurden aus der offiziellen Dokumentation des Moduls und weiterführenden Quellen kopiert und wo nötig für die Zwecke dieser Arbeit adaptiert.

# Vorbereitung der Inputdaten:

# In[1024]:


scenario_A_00 = df_final_scenario_A_00.dropna()


# In[1025]:


scenario_A_00.reset_index()


# Die Auto-Arima Funktion testet verschiedene Konfigurationen und gibt dann das beste Modell aus:

# In[1026]:


model = pm.auto_arima(scenario_A_00['LeistungenTotal'],seasonal=False,
                      start_p=0, start_q=0, max_order=4, test='adf',error_action='ignore',  
                           suppress_warnings=True,
                      stepwise=True, trace=True)


# Mit der bewährten Methodik werden die Daten in Trainings-/Testdatensatz aufgeteilt:

# In[1027]:


df_scenario_A_00_train, df_scenario_A_00_test = scenario_A_00[1:33], scenario_A_00[33:] 


# Trainieren des Modelles anhand der Trainingsdaten:

# In[1028]:


model.fit(df_scenario_A_00_train['LeistungenTotal'])


# In[1029]:


forecast=model.predict(n_periods=9, return_conf_int=True)


# Die Vorhersagen sehen wie folgt aus:

# In[1030]:


forecast


# Diese kann man einfach in ein Dataframe umwandeln:

# In[1031]:


forecast_df = pd.DataFrame(forecast[0],index = df_scenario_A_00_test.index,columns=['Prediction'])


# In[1032]:


forecast_df


# Die errechneten Werte lassen sich zusammen mit den effektiven Beobachtungen in einem Diagramm darstellen:

# In[1033]:


pd.concat([scenario_A_00['LeistungenTotal'],forecast_df],axis=1).plot()


# Das Resultat ist eine gute Annäherung. Da aber mit dem skforecast Modul im Abschnitt 1.6.1 bessere Resultate erzielt werden konnten, wird hier nicht weiter Zeit investiert.

# ## Vorbereitung Resultate (Markdown Generator)

# In[1034]:


# copy the dataset
df_1981_2021_c = df_1981_2021.copy()
# get sums for each year (population)
df_1981_2021_c['Total'] = df_1981_2021_c.iloc[:, 1:21].sum(axis=1)
df_1981_2021_c = df_1981_2021_c[['Jahr','Total','LeistungenTotal']]

# sum the population counts of the agegroups to get the total counts
df_final_scenario_A_00['Total'] = df_final_scenario_A_00.iloc[:, 1:21].sum(axis=1)
df_final_scenario_B_00['Total'] = df_final_scenario_B_00.iloc[:, 1:21].sum(axis=1)
df_final_scenario_C_00['Total'] = df_final_scenario_C_00.iloc[:, 1:21].sum(axis=1)
df_final_scenario_D_00['Total'] = df_final_scenario_D_00.iloc[:, 1:21].sum(axis=1)
df_final_scenario_A_03['Total'] = df_final_scenario_A_03.iloc[:, 1:21].sum(axis=1)

# filter for data of 2022 and later (predictions)
df_final_scenario_A_00_2022 = df_final_scenario_A_00.query('Jahr>=2022')[['Jahr','Total']]
df_final_scenario_B_00_2022 = df_final_scenario_B_00.query('Jahr>=2022')[['Jahr','Total']]
df_final_scenario_C_00_2022 = df_final_scenario_C_00.query('Jahr>=2022')[['Jahr','Total']]
df_final_scenario_D_00_2022 = df_final_scenario_D_00.query('Jahr>=2022')[['Jahr','Total']]
df_final_scenario_A_03_2022 = df_final_scenario_A_03.query('Jahr>=2022')[['Jahr','Total']]

# reset indexes for join operations
df_all_scenarios_pred_c = df_all_scenarios_pred.copy()
df_all_scenarios_pred_c = df_all_scenarios_pred_c.reset_index()
df_final_scenario_A_00_2022 = df_final_scenario_A_00_2022.reset_index()
df_final_scenario_B_00_2022 = df_final_scenario_B_00_2022.reset_index()
df_final_scenario_C_00_2022 = df_final_scenario_C_00_2022.reset_index()
df_final_scenario_D_00_2022 = df_final_scenario_D_00_2022.reset_index()
df_final_scenario_A_03_2022 = df_final_scenario_A_03_2022.reset_index()

# join the predicted total cost values
df_final_scenario_A_00_2022['LeistungenTotal'] = df_all_scenarios_pred_c[['A_00_Predicted']]
df_final_scenario_B_00_2022['LeistungenTotal'] = df_all_scenarios_pred_c[['B_00_Predicted']]
df_final_scenario_C_00_2022['LeistungenTotal'] = df_all_scenarios_pred_c[['C_00_Predicted']]
df_final_scenario_D_00_2022['LeistungenTotal'] = df_all_scenarios_pred_c[['D_00_Predicted']]
df_final_scenario_A_03_2022['LeistungenTotal'] = df_all_scenarios_pred_c[['A_03_Predicted']]

# calculate values per capita
df_1981_2021_c['proKopf'] = df_1981_2021_c['LeistungenTotal'] / df_1981_2021_c['Total']
df_final_scenario_A_00_2022['proKopf'] = df_final_scenario_A_00_2022['LeistungenTotal'] / df_final_scenario_A_00_2022['Total']
df_final_scenario_B_00_2022['proKopf'] = df_final_scenario_B_00_2022['LeistungenTotal'] / df_final_scenario_B_00_2022['Total']
df_final_scenario_C_00_2022['proKopf'] = df_final_scenario_C_00_2022['LeistungenTotal'] / df_final_scenario_C_00_2022['Total']
df_final_scenario_D_00_2022['proKopf'] = df_final_scenario_D_00_2022['LeistungenTotal'] / df_final_scenario_D_00_2022['Total']
df_final_scenario_A_03_2022['proKopf'] = df_final_scenario_A_03_2022['LeistungenTotal'] / df_final_scenario_A_03_2022['Total']


# In[1035]:


df_results = df_1981_2021_c.query('Jahr==2020')

print("**Szenario Referenzszenario A-00-2020**")
print("")
df_results_A_00 = df_results.copy()
for x in range(3,8):
    y = 2000 + x*10
    df_results_A_00 = pd.concat([df_results_A_00,
                        pd.DataFrame([(y,
                         df_final_scenario_A_00_2022
                            .loc[df_final_scenario_A_00_2022['Jahr'] == y]['Total'],
                         df_final_scenario_A_00_2022
                            .loc[df_final_scenario_A_00_2022['Jahr'] == y]['LeistungenTotal'],
                         df_final_scenario_A_00_2022
                            .loc[df_final_scenario_A_00_2022['Jahr'] == y]['proKopf'])],
                   columns=['Jahr', 'Total','LeistungenTotal','proKopf'])])

df_results_A_00 = df_results_A_00.reset_index(drop=True)

print(df_results_A_00.to_markdown())
print("")
print("")

print("**Szenario 'hohes' Szenario B-00-2020**")
print("")
df_results_B_00 = df_results.copy()
for x in range(3,8):
    y = 2000 + x*10
    df_results_B_00 = pd.concat([df_results_B_00,
                        pd.DataFrame([(y,
                         df_final_scenario_B_00_2022
                            .loc[df_final_scenario_B_00_2022['Jahr'] == y]['Total'],
                         df_final_scenario_B_00_2022
                            .loc[df_final_scenario_B_00_2022['Jahr'] == y]['LeistungenTotal'],
                         df_final_scenario_B_00_2022
                            .loc[df_final_scenario_B_00_2022['Jahr'] == y]['proKopf'])],
                   columns=['Jahr', 'Total','LeistungenTotal','proKopf'])])

df_results_B_00 = df_results_B_00.reset_index(drop=True)

print(df_results_B_00.to_markdown())
print("")
print("")

print("**Szenario 'tiefes' Szenario C-00-2020**")
print("")
df_results_C_00 = df_results.copy()
for x in range(3,8):
    y = 2000 + x*10
    df_results_C_00 = pd.concat([df_results_C_00,
                        pd.DataFrame([(y,
                         df_final_scenario_C_00_2022
                            .loc[df_final_scenario_C_00_2022['Jahr'] == y]['Total'],
                         df_final_scenario_C_00_2022
                            .loc[df_final_scenario_C_00_2022['Jahr'] == y]['LeistungenTotal'],
                         df_final_scenario_C_00_2022
                            .loc[df_final_scenario_C_00_2022['Jahr'] == y]['proKopf'])],
                   columns=['Jahr', 'Total','LeistungenTotal','proKopf'])])

df_results_C_00 = df_results_C_00.reset_index(drop=True)

print(df_results_C_00.to_markdown())
print("")
print("")

print("**Szenario 'verstärkte Alterung' D-00-2020**")
print("")
df_results_D_00 = df_results.copy()
for x in range(3,8):
    y = 2000 + x*10
    df_results_D_00 = pd.concat([df_results_D_00,
                        pd.DataFrame([(y,
                         df_final_scenario_D_00_2022
                            .loc[df_final_scenario_D_00_2022['Jahr'] == y]['Total'],
                         df_final_scenario_D_00_2022
                            .loc[df_final_scenario_D_00_2022['Jahr'] == y]['LeistungenTotal'],
                         df_final_scenario_D_00_2022
                            .loc[df_final_scenario_D_00_2022['Jahr'] == y]['proKopf'])],
                   columns=['Jahr', 'Total','LeistungenTotal','proKopf'])])

df_results_D_00 = df_results_D_00.reset_index(drop=True)

print(df_results_D_00.to_markdown())
print("")
print("")

print("**Szenario 'höhere Lebenserwartung bei der Geburt' A-03-2020**")
print("")
df_results_A_03 = df_results.copy()
for x in range(3,8):
    y = 2000 + x*10
    df_results_A_03 = pd.concat([df_results_A_03,
                        pd.DataFrame([(y,
                         df_final_scenario_A_03_2022
                            .loc[df_final_scenario_A_03_2022['Jahr'] == y]['Total'],
                         df_final_scenario_A_03_2022
                            .loc[df_final_scenario_A_03_2022['Jahr'] == y]['LeistungenTotal'],
                         df_final_scenario_A_03_2022
                            .loc[df_final_scenario_A_03_2022['Jahr'] == y]['proKopf'])],
                   columns=['Jahr', 'Total','LeistungenTotal','proKopf'])])

df_results_A_03 = df_results_A_03.reset_index(drop=True)

print(df_results_A_03.to_markdown())


# ## Resultate

# Die Resultate des skforecast Modelles mit dem Ridge Regressor, welcher besonders gut ist für Szenarien, in denen die unabhängigen Variablen stark korrelieren, sollen hier nochmal detailliert aufgeführt werden. Ein Vorstellen aller Resultate von allen getesteten Modellen würde den Rahmen dieser Arbeit sprengen, deswegen wird darauf verzichtet.

# Mit einem Trainings-/Testdatensplit von 90% und 10%, konnten unter jeweils der Berücksichtigung der letzten 10 Beobachtungen/Schritte in der Zeitreihe, die folgenden Resultate erzielt werden:

# * Mittlerer absoluter Fehler zwischen Testset und Vorhersagen: 892'479'060.9
# * Durchschnittliche totale Kosten der Observationen im Testet: 83'340'886'000.0
# * Somit resultierte eine mittlere Abweichung von 1.07%

# Grafisch dargestellt sieht man in blau die effektiven Beobachtungen (Trainingsdaten), in gelb die Testdaten und in grün die errechnete Entwicklung der Kosten:

# ![Alt-Text](./img/result_reference_scenario.png "skforecast: Resultat für Referenzszenario")

# Das Modell wurde in der Folge genutzt, um für die verschiedenen Bevölkerungsszenarien vom Bund die totalen Kosten bis 2070 vorherzusagen. Die errechnete Entwicklung der Gesamtbevölkerung in der Schweiz ist nachfolgend grafisch dargestellt.

# ![Alt-Text](./img/szenarien_bevoelkerungsentwicklung.png "Szenarien Bevölkerungsentwicklung")

# Anhand der Anzahl Personen in den verschiedenen Altersgruppen, wurden dann die gesamten Gesundheitskosten pro Szenario und Jahr bis 2070 vorhergesagt. Aus Platzgründen wurden diese Resultate hier gekürzt und es werden nur jeweils die Werte pro Szenario alle 10 Jahre gezeigt.

# **Szenario Referenzszenario A-00-2020**
#   
# |    |   Jahr |       Total |   LeistungenTotal |   proKopf |
# |---:|-------:|------------:|------------------:|----------:|
# |  0 |   2020 | 8.6703e+06  |       8.33108e+10 |   9608.75 |
# |  1 |   2030 | 9.4308e+06  |       1.20077e+11 |  12732.4  |
# |  2 |   2040 | 1.00154e+07 |       1.5463e+11  |  15439.2  |
# |  3 |   2050 | 1.04406e+07 |       1.93766e+11 |  18558.8  |
# |  4 |   2060 | 1.08025e+07 |       2.34774e+11 |  21733.4  |
# |  5 |   2070 | 1.11386e+07 |       2.79796e+11 |  25119.6  | 

# **Szenario 'hohes' Szenario B-00-2020**
# 
# |    |   Jahr |       Total |   LeistungenTotal |   proKopf |
# |---:|-------:|------------:|------------------:|----------:|
# |  0 |   2020 | 8.6703e+06  |       8.33108e+10 |   9608.75 |
# |  1 |   2030 | 9.67221e+06 |       1.28269e+11 |  13261.6  |
# |  2 |   2040 | 1.05726e+07 |       1.79882e+11 |  17013.9  |
# |  3 |   2050 | 1.13857e+07 |       2.44716e+11 |  21493.4  |
# |  4 |   2060 | 1.21725e+07 |       3.23529e+11 |  26578.7  |
# |  5 |   2070 | 1.29555e+07 |       4.23719e+11 |  32705.6  |

# **Szenario 'tiefes' Szenario C-00-2020**
#   
# |    |   Jahr |       Total |   LeistungenTotal |   proKopf |
# |---:|-------:|------------:|------------------:|----------:|
# |  0 |   2020 | 8.6703e+06  |       8.33108e+10 |   9608.75 |
# |  1 |   2030 | 9.18938e+06 |       1.11756e+11 |  12161.5  |
# |  2 |   2040 | 9.4635e+06  |       1.28718e+11 |  13601.5  |
# |  3 |   2050 | 9.51693e+06 |       1.41114e+11 |  14827.7  |
# |  4 |   2060 | 9.48463e+06 |       1.4337e+11  |  15116    |
# |  5 |   2070 | 9.41886e+06 |       1.33595e+11 |  14183.8  |

# **Szenario 'verstärkte Alterung' D-00-2020**
#   
# |    |   Jahr |       Total |   LeistungenTotal |   proKopf |
# |---:|-------:|------------:|------------------:|----------:|
# |  0 |   2020 | 8.6703e+06  |       8.33108e+10 |   9608.75 |
# |  1 |   2030 | 9.27124e+06 |       1.11957e+11 |  12075.7  |
# |  2 |   2040 | 9.63767e+06 |       1.29383e+11 |  13424.7  |
# |  3 |   2050 | 9.7874e+06  |       1.42785e+11 |  14588.7  |
# |  4 |   2060 | 9.83707e+06 |       1.46084e+11 |  14850.3  |
# |  5 |   2070 | 9.8139e+06  |       1.37399e+11 |  14000.4  |

# **Szenario 'höhere Lebenserwartung bei der Geburt' A-03-2020**
#   
# |    |   Jahr |       Total |   LeistungenTotal |   proKopf |
# |---:|-------:|------------:|------------------:|----------:|
# |  0 |   2020 | 8.6703e+06  |       8.33108e+10 |   9608.75 |
# |  1 |   2030 | 9.47156e+06 |       1.20173e+11 |  12687.8  |
# |  2 |   2040 | 1.01029e+07 |       1.54953e+11 |  15337.5  |
# |  3 |   2050 | 1.05779e+07 |       1.94568e+11 |  18393.9  |
# |  4 |   2060 | 1.09844e+07 |       2.36062e+11 |  21490.7  |
# |  5 |   2070 | 1.13453e+07 |       2.81582e+11 |  24819.2  |

# Im hohen Szenario droht der Schweiz also ein Anstieg von 2020 noch 83 Mia. CHF auf 2070 insgesamt 424 Mia. CHF. Die pro Kopf Kosten steigen dabei um 23'000 CHF.
#   
# Das tiefe Szenario zeigt einen Anstieg von 2020 noch 83 Mia. CHF auf 2070 insgesamt 134 Mia. CHF. Die pro Kopf Kosten steigen hier um 4'500 CHF.
# 
# _(Die genauen Werte sind den obigen Tabellen zu entnehmen, sie wurden an dieser Stelle gerundet)_

# Die Kostenentwicklungen für jedes Szenario zeigen grafisch dargestellt eine ähnliche Entwicklung wie bei der Gesamtbevölkerung, allerdings ist beim hohen Szenario ein fast schon exponentielles Wachstum ersichtlich.

# ![Alt-Text](./img/result_all_scenarios.png "skforecast: Resultat für alle Szenarien")

# ## Diskussion

# Die Analyse und die Vorhersage von Timeseries Daten ist, insbesondere wenn es sich um nicht-stationäre Daten handelt, sehr schwierig. Die meisten der ausprobierten Modelle sind optimiert für stationäre Daten und bräuchten vor Allem auch viel mehr Inputdaten. Somit war die Performance der einzelnen Modelle für mich persönlich sehr unbefriedigend. Das skforecast Modul, welches die besten Resultate liefern konnte und insgesamt auch am einfachsten anzuwenden war, hat mich aber sehr überzeugt. Es zeigt für mich wieder mal eindrücklich, dass die grosse Community im Data Science Bereich einfach auch kollaborativ tolle Module/Frameworks erarbeiten kann, die dann hoffentlich irgendwann in die Top-Level Projekte (z.B. sklearn) integriert werden. 
# 
# Im Folgenden nochmal die für mich wichtigsten Erkenntnisse:
# * Der grösste Teil des Aufwands liegt ganz klar im Beschaffen, Aufbereiten und dem Zusammenführen der Inputdaten.
# * Datensets mit <50 Zeilen sind trotz grosser Anzahl Features einfach oft zu klein, um ansprechende Resultate zu erzielen.
# * Wünschenswert wären Datensets mit weniger Features (<15), aber massenhaft Beobachtungen.
# * Es gibt zwar sehr viele Modelle die mittlerweile mit tiefer Eintrittsschwelle verwendet/adaptiert werden können, sie dann aber richtig im eigenen Kontext anzuwenden ist nochmal eine ganz andere Geschichte. Aus diesem Grund lohnt es sich, mit den einfacheren Modellen zu starten, und nur wenn mit diesen nicht die gewünschten Resultate erzielt werden konnten, zu den komplexeren Modellen zu wechseln.
# * Für viele Modelle wäre es hilfreich gewesen, wenn das CAS Statistische Datenanalyse und Datenvisualisierung im Vorfeld besucht worden wäre. Damit man Anhaltspunkte bekommt, ob das Modell überhaupt erfolgreich auf die Daten angewendet werden kann, sind oft die Standard-Statistik-Tests notwendig (z.B. Prüfung stationär vs nicht-stationär, Abhängigkeiten zwischen Variablen, etc.).
# * Timeseries Daten sind eine Sache für sich, und eine vertiefte Behandlung im Kurs CAS Machine Learning wäre wünschenswert und sinnvoll.
# * Die konstruierten Fallbeispiele aus dem Kurs kommen oft mit der Standardimplementation der Modelle in keras oder sklearn aus. Die real-life Szenarien hingegen erfordern meist Hilfsfunktionen/-methoden und andere Tweaks.
# * Die Skalierung von Inputdaten und das Rück-Skalieren liefern nicht 1:1 dieselben Resultate, da es nur stochastische Prozesse sind.
# * Wie bei Software-Entwicklungsprojekten lohnt es sich auch hier, am Anfang einen Plan zu machen wie man vorgehen möchte, und auch neben dem effektiven Coding viel zu dokumentieren und zu modellieren/zeichnen. Das nimmt zwar viel Zeit in Anspruch, hilft aber sehr, da man sich so auch einige Umwege ersparen kann.
# * Insgesamt konnte ich im Laufe dieser Arbeit wahnsinnig viel dazulernen, vor Allem auch in der effizienten und nachhaltigen Prozessierung von Daten innerhalb von Jupyter Notebook. Die Recherchen über all die ausprobierten Modelle für Timeseries Daten haben zwar viel Zeit in Anspruch genommen, aber gleichzeitig auch viel Spass gemacht. Es wäre sicher ein gutes Thema für eine Masterarbeit.
