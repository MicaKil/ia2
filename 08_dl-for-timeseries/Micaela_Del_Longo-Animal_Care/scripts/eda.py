import kaggle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# cargamos el dataset
df = pd.read_csv('../micaela_data/animal-data-1.csv')

print("Tipos de Datos\n", df.dtypes, "\n")

shape = df.shape
print("Cantidad de Filas: %d\nCantidad de Columnas: %d\n" % (shape[0], shape[1]))

print("Duplicados: %d" % df.duplicated().sum())

# list speciesnames
species = df['speciesname'].unique()
species_counts = df['speciesname'].value_counts()

print("Cantidad de Especies: ", len(species))
print("Epecies: ", species)

species = species_counts.index
sizes = species_counts.values / species_counts.values.sum() * 100
colors = plt.cm.Paired(range(len(species)))
plt.pie(sizes, labels=species, colors=colors, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Species Distribution')
plt.show()
