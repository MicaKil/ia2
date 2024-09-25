# import kaggle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

# Deep Learning para Series de Tiempo
# Resolver un problema de predicción de series de tiempo para predecir las mascotas que recibirá un refugio de animales
# domésticos. Dividir entre perros y gatos.
# **Estudiante:** Micaela Del Longo
# **Dataset:** https://www.kaggle.com/datasets/jinbonnie/animal-data

# cargamos el dataset
df = pd.read_csv('../micaela_data/animal-data-1.csv')

df = df.rename(
	columns={
		'id': 'ID',
		'intakedate': 'Intake Date',
		'intakereason': 'Intake Reason',
		'istransfer': 'Is Transfer',
		'sheltercode': 'Shelter Code',
		'identichipnumber': 'Identi Chip Number',
		'animalname': 'Animal Name',
		'breedname': 'Breed Name',
		'basecolour': 'Base Colour',
		'speciesname': 'Species Name',
		'animalage': 'Animal Age',
		'sexname': 'Sex Name',
		'location': 'Location',
		'movementdate': 'Movement Date',
		'movementtype': 'Movement Type',
		'istrial': 'Is Trial',
		'returndate': 'Return Date',
		'returnedreason': 'Returned Reason',
		'deceaseddate': 'Deceased Date',
		'deceasedreason': 'Deceased Reason',
		'diedoffshelter': 'Died Off Shelter',
		'puttosleep': 'Put to Sleep',
		'isdoa': 'Is DOA'
	}
)

print("Tipos de Datos")
print(df.dtypes)

shape = df.shape
print("\nCantidad de Filas: %d\nCantidad de Columnas: %d\n" % (shape[0], shape[1]))

print("Duplicados: %d" % df.duplicated().sum())

# number of uniques ID
print("ID Unicos: %d" % df['ID'].nunique())

# drop all species except 'Cat' and 'Dog'
df = df[df['Species Name'].isin(['Cat', 'Dog'])]
