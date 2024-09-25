import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def transform_data(dataframe: pd.DataFrame) -> pd.DataFrame:
	"""
	Transform the dataframe to a format that can be used for time series analysis
	:param dataframe: The dataframe to transform
	:return: The transformed dataframe
	"""

	# Tomo las columnas de interés para el análisis de series de tiempo: 'Intake DateTime', 'Movement DateTime', 'Return DateTime' y 'Deceased DateTime'.

	df_intake = dataframe[['Intake DateTime', 'Species Name', 'ID']].copy()
	df_intake = df_intake.drop_duplicates()  # eliminar las filas duplicadas

	df_movement = dataframe[['Movement DateTime', 'Species Name', 'ID']].copy()
	df_movement = df_movement.drop_duplicates()
	df_movement = df_movement.dropna()

	df_return = dataframe[['Return DateTime', 'Species Name', 'ID']].copy()
	df_return = df_return.drop_duplicates()
	df_return = df_return.dropna()

	df_deceased = dataframe[['Deceased DateTime', 'Species Name', 'ID']].copy()
	df_deceased = df_deceased.drop_duplicates()
	df_deceased = df_deceased.dropna()

	# Calculo la cantidad de perros y gatos que ingresaron/egresaron/fueron retornados/fallecieron por día.
	df_grouped_intake = df_intake.groupby(['Intake DateTime', 'Species Name']).size().unstack().fillna(0)
	df_grouped_intake.columns = df_grouped_intake.rename(columns={'Cat': 'Intake Cats', 'Dog': 'Intake Dogs'}).columns

	df_grouped_movement = df_movement.groupby(['Movement DateTime', 'Species Name']).size().unstack().fillna(0)
	df_grouped_movement.columns = df_grouped_movement.rename(columns={'Cat': 'Moved Cats', 'Dog': 'Moved Dogs'}).columns

	df_grouped_return = df_return.groupby(['Return DateTime', 'Species Name']).size().unstack().fillna(0)
	df_grouped_return.columns = df_grouped_return.rename(
		columns={'Cat': 'Returned Cats', 'Dog': 'Returned Dogs'}).columns

	df_grouped_deceased = df_deceased.groupby(['Deceased DateTime', 'Species Name']).size().unstack().fillna(0)
	df_grouped_deceased.columns = df_grouped_deceased.rename(
		columns={'Cat': 'Deceased Cats', 'Dog': 'Deceased Dogs'}).columns

	# Agrupo a todos los registros en un solo DataFrame

	# Reseteo los índices para poder concatenar los DataFrames en la fecha
	df_grouped_intake_reset = df_grouped_intake.reset_index()
	df_grouped_movement_reset = df_grouped_movement.reset_index()
	df_grouped_return_reset = df_grouped_return.reset_index()
	df_grouped_deceased_reset = df_grouped_deceased.reset_index()

	# Renombro las columnas para que tengan el mismo nombre
	df_grouped_intake_rename = df_grouped_intake_reset.rename(columns={'Intake DateTime': 'Date'})
	df_grouped_movement_rename = df_grouped_movement_reset.rename(columns={'Movement DateTime': 'Date'})
	df_grouped_return_rename = df_grouped_return_reset.rename(columns={'Return DateTime': 'Date'})
	df_grouped_deceased_rename = df_grouped_deceased_reset.rename(columns={'Deceased DateTime': 'Date'})

	df_grouped = pd.merge(df_grouped_intake_rename, df_grouped_movement_rename, on='Date', how='outer')
	df_grouped = pd.merge(df_grouped, df_grouped_return_rename, on='Date', how='outer')
	df_grouped = pd.merge(df_grouped, df_grouped_deceased_rename, on='Date', how='outer')

	df_grouped = df_grouped.fillna(0)
	df_grouped = df_grouped.set_index('Date', drop=True)
	df_grouped.index = pd.to_datetime(df_grouped.index)

	return df_grouped


def plot_series(dataframe: pd.DataFrame, title: str):
	"""
	Plot the series of the dataframe
	:param dataframe: The dataframe to plot
	:param title: The title of the plot
	:return: None
	"""
	sns.set_theme(style='whitegrid', palette='Set2')

	plt.figure(figsize=(12, 8))
	sns.lineplot(data=dataframe, dashes=False)
	plt.title(title)
	plt.xlabel('Date')
	plt.ylabel('Quantity')
	plt.show()


def plot_category(dataframe: pd.DataFrame, category: str, title: str):
	"""
	Plot a single category from the dataframe
	:param dataframe: The dataframe to plot
	:param category: The category to plot
	:param title: The title of the plot
	:return: None
	"""
	sns.set_theme(style='whitegrid', palette='Set2')

	plt.figure(figsize=(12, 8))
	sns.lineplot(data=dataframe, y=category, x=dataframe.index)
	plt.title(title)
	plt.xlabel('Date')
	plt.ylabel('Quantity')
	plt.show()


def plot_cat_categories(dataframe: pd.DataFrame):
	"""
	Plot the categories related to cats from the dataframe
	:param dataframe: The dataframe to plot
	:return: None
	"""
	sns.set_theme(style='whitegrid', palette='Set2')

	fig, axs = plt.subplots(2, 2, figsize=(12, 8))
	sns.lineplot(data=dataframe, y='Intake Cats', x=dataframe.index, ax=axs[0, 0])
	sns.lineplot(data=dataframe, y='Moved Cats', x=dataframe.index, ax=axs[0, 1])
	sns.lineplot(data=dataframe, y='Returned Cats', x=dataframe.index, ax=axs[1, 0])
	sns.lineplot(data=dataframe, y='Deceased Cats', x=dataframe.index, ax=axs[1, 1])
	plt.tight_layout()
	plt.show()


def plot_dog_categories(dataframe: pd.DataFrame):
	"""
	Plot the categories regarding dogs of the dataframe
	:param dataframe: The dataframe to plot
	:return: None
	"""
	sns.set_theme(style='whitegrid', palette='Set2')

	fig, axs = plt.subplots(2, 2, figsize=(12, 8))
	sns.lineplot(data=dataframe, y='Intake Dogs', x=dataframe.index, ax=axs[0, 0])
	sns.lineplot(data=dataframe, y='Moved Dogs', x=dataframe.index, ax=axs[0, 1])
	sns.lineplot(data=dataframe, y='Returned Dogs', x=dataframe.index, ax=axs[1, 0])
	sns.lineplot(data=dataframe, y='Deceased Dogs', x=dataframe.index, ax=axs[1, 1])
	plt.title('Dogs')
	plt.tight_layout()
	plt.show()

def boxplot(dataframe: pd.DataFrame, title: str):
	"""
	Plot the boxplot of the dataframe
	:param dataframe: The dataframe to plot
	:param title: The title of the plot
	:return: None
	"""
	sns.set_theme(style='whitegrid', palette='Set2')

	plt.figure(figsize=(12, 8))
	sns.boxplot(data=dataframe, orient='v')

	plt.xlabel('Category')
	plt.ylabel('Quantity')
	plt.show()

def compare_boxplots(df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame, category: str, title:str):
	"""
	Compare the boxplots of the three dataframes
	:param df1: The first dataframe
	:param df2: The second dataframe
	:param df3: The third dataframe
	:param category: The category to compare
	:param title: The title of the plot
	:return: None
	"""

	sns.set_theme(style='whitegrid', palette='Set2')

	fig, axs = plt.subplots(1, 3, figsize=(12, 8))
	sns.boxplot(data=df1, y=category, orient='v', ax=axs[0])
	axs[0].set_title('Original')
	axs[0].set_ylabel('Quantity')
	sns.boxplot(data=df2, y=category, orient='v', ax=axs[1])
	axs[1].set_title('Daily')
	axs[1].set_ylabel('Quantity')
	sns.boxplot(data=df3, y=category, orient='v', ax=axs[2])
	axs[2].set_title('Weekly')
	axs[2].set_ylabel('Quantity')

	plt.tight_layout()
	plt.show()
