import kaggle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

# Deep Learning para Series de Tiempo
# Resolver un problema de predicción de series de tiempo para predecir las mascotas que recibirá un refugio de animales domésticos. Dividir entre perros y gatos.
# **Estudiante:** Micaela Del Longo
# **Dataset:** https://www.kaggle.com/datasets/jinbonnie/animal-data

# cargamos el dataset
df = pd.read_csv('../micaela_data/animal-data-1.csv')

df = df.rename(
	columns={
		'id': 'ID', 'intakedate': 'Intake Date', 'intakereason': 'Intake Reason', 'istransfer': 'Is Transfer',
		'sheltercode': 'Shelter Code', 'identichipnumber': 'Identi Chip Number', 'animalname': 'Animal Name',
		'breedname': 'Breed Name', 'basecolour': 'Base Colour', 'speciesname': 'Species Name',
		'animalage': 'Animal Age', 'sexname': 'Sex Name', 'location': 'Location', 'movementdate': 'Movement Date',
		'movementtype': 'Movement Type', 'istrial': 'Is Trial', 'returndate': 'Return Date',
		'returnedreason': 'Returned Reason', 'deceaseddate': 'Deceased Date', 'deceasedreason': 'Deceased Reason',
		'diedoffshelter': 'Died Off Shelter', 'puttosleep': 'Put to Sleep', 'isdoa': 'Is DOA'
	}
)

print("Tipos de Datos\n", df.dtypes, "\n")

shape = df.shape
print("Cantidad de Filas: %d\nCantidad de Columnas: %d\n" % (shape[0], shape[1]))

print("Duplicados: %d" % df.duplicated().sum())

# list speciesnames
species = df['Species Name'].unique()
species_counts = df['Species Name'].value_counts()
species_quantity = len(species)
print("Cantidad de Especies: ", species_quantity)
print("Especies: ", species)

species = species_counts.index  # species names
sizes = species_counts.values / species_counts.values.sum() * 100  # species distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))  # Create a figure and a set of subplots

bars = ax1.bar(species, sizes, color=sns.color_palette("hls", species_quantity))
ax1.set_xlabel('Species')
ax1.set_ylabel('Percentage')
ax1.set_title('Species Distribution')
ax1.set_xticks(species)
ax1.set_xticklabels(species, rotation=90)

ax2.axis('off')  # Turn off the axis
ax2.legend(bars, [f"{species[i]}: {sizes[i]:.3f}%" for i in range(species_quantity)], loc="center")

plt.tight_layout()

# plt.savefig('../micaela_data/figs/eda/bar_plot-species_distribution.png')
plt.show()


