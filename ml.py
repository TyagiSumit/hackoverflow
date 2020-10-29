import pandas as pd
import warnings
import csv
import pickle
warnings.filterwarnings('ignore')

# loading database of hospitals
df = pd.read_csv('data/nin-health-facilities.csv', engine='python')
df = df[ df.latitude != '\\N'][df.longitude != '\\N']


# loading total symptoms

with open('data/total_symp.csv', newline='') as f:
    reader = csv.reader(f)
    total_symp = list(reader)


# load model

loaded_model = pickle.load(open('models/model', 'rb'))


# nearest hospital

def shortest_entry(lat, long):
    dis = ( (df.latitude.astype('float') - lat )**2 + (df.longitude.astype('float') - long )**2 )** 0.5
    return dict(zip(df.columns, df.loc[dis.argmin()].values))



def prepare_data(symptoms):
    symptoms.replace('+', ' ')
    symptoms = symptoms.split(',')
    return [[ 1 if x in symptoms else 0 for x in total_symp[0]]]

def predict(symptoms):
    symp = prepare_data(symptoms)
    return loaded_model.predict(symp)

