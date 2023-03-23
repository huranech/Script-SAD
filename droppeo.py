import pandas as pd

# Cargar el archivo CSV en un dataframe
df = pd.read_csv('datasetForTheExam_SubGrupo1.csv')

# Eliminar la columna 'Target'
df = df.drop('Class', axis=1)

# Guardar el dataframe en un nuevo archivo CSV
df.to_csv('EjemploExamen_sinTarget.csv', index=False)