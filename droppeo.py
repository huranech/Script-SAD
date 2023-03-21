import pandas as pd

# Cargar el archivo CSV en un dataframe
df = pd.read_csv('iris.csv')

# Eliminar la columna 'Target'
df = df.drop('Especie', axis=1)

# Guardar el dataframe en un nuevo archivo CSV
df.to_csv('iris_sin_target.csv', index=False)