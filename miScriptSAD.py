import imblearn
import sklearn
import pandas as pd
import pickle
import sys
import numpy as np
import getopt

# Runear el código. Pide que se le pasen argumentos
if __name__ == '__main__':
    print('ARGV   :',sys.argv[1:])
    try:
        options,remainder = getopt.getopt(sys.argv[1:],'p:m:f:h',['path=','model=','testFile=','h'])
    except getopt.GetoptError as err:
        print('ERROR:',err)
        sys.exit(1)
    print('OPTIONS   :',options)

    for opt,arg in options:
        if opt in ('-p','--path'):
            p = arg
        elif opt in ('-f', '--file'):
            f = arg
        elif opt in ('-m', '--model'):
            m = arg
        elif opt in ('-h','--help'):
            print(' -p modelAndTestFilePath \n -m modelFileName -f testFileName\n ')
            exit(1)

    if p == './':
        model=p+str(m)
        iFile = p+ str(f)
    else:
        model=p+"/"+str(m)
        iFile = p+"/" + str(f)


    # función que tiene que ver con la codificación en utf-8
    def coerce_to_unicode(x):
        if sys.version_info < (3, 0):
            if isinstance(x, str):
                return unicode(x, 'utf-8')
            else:
                return unicode(x)
        else:
            return str(x)
        
    # abrir el fichero .csv y cargarlo en un dataframe de pandas
    ml_dataset = pd.read_csv(iFile)

    # [HARDCODE] seleccionar únicamente los features que nos interesan 
    ml_dataset = ml_dataset[['Especie', 'Ancho de sepalo', 'Largo de sepalo', 'Largo de petalo', 'Ancho de petalo']]

    # [HARDCODE] pasamos los valores categoriales y de texto a unicode y los numéricos a float
    categorical_features = []
    numerical_features = ['Ancho de sepalo', 'Largo de sepalo', 'Largo de petalo', 'Ancho de petalo']
    text_features = []

    for feature in categorical_features:  # valores categoriales
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)
    for feature in text_features:  # valores de tipo texto
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)
    for feature in numerical_features:  # valores numéricos
        if ml_dataset[feature].dtype == np.dtype('M8[ns]') or (hasattr(ml_dataset[feature].dtype, 'base') and ml_dataset[feature].dtype.base == np.dtype('M8[ns]')):
            ml_dataset[feature] = datetime_to_epoch(ml_dataset[feature])
        else:
            ml_dataset[feature] = ml_dataset[feature].astype('double')
    
    # [HARDCODE] creamos la columna TARGET con los valores
    target_map = {'Iris-versicolor': 0, 'Iris-virginica': 1, 'Iris-setosa': 2}
    ml_dataset['__target__'] = ml_dataset['Especie'].map(str).map(target_map)
    del ml_dataset['Especie']

    