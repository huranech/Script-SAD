import imblearn
import sklearn
import pandas as pd
import pickle
import sys
import numpy as np
import getopt
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Runear el código. Pide que se le pasen argumentos
if __name__ == '__main__':
    print('ARGV   :',sys.argv[1:])
    try:
        options,remainder = getopt.getopt(sys.argv[1:],'p:m:f:h:k:',['path=','model=','testFile=','h','k='])
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
        elif opt in ('-k','--kparameter'):
            k = int(arg)

    if p == './':
        # model=p+str(m)
        iFile = p+ str(f)
    else:
        # model=p+"/"+str(m)
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

    # se eliminan las filas para las que el TARGET es null / se pasan los datos que fueran float a Integer
    ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]
    ml_dataset['__target__'] = ml_dataset['__target__'].astype(np.int64)

    # se separa la muestra en los conjuntos train y test donde el test representa el 20% y la proporción de targets es la misma para todo
    train, test = train_test_split(ml_dataset,test_size=0.2,random_state=42,stratify=ml_dataset[['__target__']])

    # [HARDCODE] se escoge la forma en la que se van a tratar los valores faltantes
    drop_rows_when_missing = []
    impute_when_missing = [{'feature': 'Ancho de sepalo', 'impute_with': 'MEAN'}, {'feature': 'Largo de sepalo', 'impute_with': 'MEAN'}, {'feature': 'Largo de petalo', 'impute_with': 'MEAN'}, {'feature': 'Ancho de petalo', 'impute_with': 'MEAN'}]

    # se borran las filas donde hay datos faltantes para las features en 'drop_rows_when_missing'
    for feature in drop_rows_when_missing:
        train = train[train[feature].notnull()]
        test = test[test[feature].notnull()]
        print ('Dropped missing records in %s' % feature)

    # se imputan los datos para las filas donde hay datos faltantes para las features dentro de 'impute_when_missing'
    for feature in impute_when_missing:
        if feature['impute_with'] == 'MEAN':
            v = train[feature['feature']].mean()
        elif feature['impute_with'] == 'MEDIAN':
            v = train[feature['feature']].median()
        elif feature['impute_with'] == 'CREATE_CATEGORY':
            v = 'NULL_CATEGORY'
        elif feature['impute_with'] == 'MODE':
            v = train[feature['feature']].value_counts().index[0]
        elif feature['impute_with'] == 'CONSTANT':
            v = feature['value']
        train[feature['feature']] = train[feature['feature']].fillna(v)
        test[feature['feature']] = test[feature['feature']].fillna(v)
        print ('Imputed missing values in feature %s with value %s' % (feature['feature'], coerce_to_unicode(v)))

    # se reescalan los valores de las features con una media de 0 y una desviación estándar de 1
    rescale_features = {'Ancho de sepalo': 'AVGSTD', 'Largo de sepalo': 'AVGSTD', 'Largo de petalo': 'AVGSTD', 'Ancho de petalo': 'AVGSTD'}
    for (feature_name, rescale_method) in rescale_features.items():
        if rescale_method == 'MINMAX':
            _min = train[feature_name].min()
            _max = train[feature_name].max()
            scale = _max - _min
            shift = _min
        else:
            shift = train[feature_name].mean()
            scale = train[feature_name].std()
        if scale == 0.:
            del train[feature_name]
            del test[feature_name]
            print ('Feature %s was dropped because it has no variance' % feature_name)
        else:
            print ('Rescaled %s' % feature_name)
            train[feature_name] = (train[feature_name] - shift).astype(np.float64) / scale
            test[feature_name] = (test[feature_name] - shift).astype(np.float64) / scale

    # se separan las partes "feature (X)" de las partes "label (Y)" en los conjuntos train y test
    X_train = train.drop('__target__', axis=1)
    X_test = test.drop('__target__', axis=1)

    Y_train = np.array(train['__target__'])
    Y_test = np.array(test['__target__'])

    # conjunto de bucles donde sucede el barrido de hiperparámetros. Primero llenamos arrays con su respectivo parámetro
    barridoK = []
    for numero in range(k):
        if numero == 0:
            # no ocurre nada, no se permite el valor 0
        elif numero % 2 == 0:
            
    
    for parametroK in barridoK:

        # [HARDCODE] se crea el modelo con unos hiperparámetros predefinidos
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=parametroK,
                            weights='uniform',
                            algorithm='auto',
                            leaf_size=30,
                            p=2)
        
        # se balancean los datos (esto puede no interesarnos)
        clf.class_weight = "balanced"

        # entrena el algoritmo para que, basándose en los datos de los features de X_train se cree una coincidencia con las labels de y_train
        clf.fit(X_train, Y_train)

        # se realizan las predicciones
        predictions = clf.predict(X_test)
        probas = clf.predict_proba(X_test)

        predictions = pd.Series(data=predictions, index=X_test.index, name='predicted_value')
        cols = [
            u'probability_of_value_%s' % label
            for (_, label) in sorted([(int(target_map[label]), label) for label in target_map])
        ]
        probabilities = pd.DataFrame(data=probas, index=X_test.index, columns=cols)

        # construir la evaluación de los resultados
        results_test = X_test.join(predictions, how='left')
        results_test = results_test.join(probabilities, how='left')
        results_test = results_test.join(test['__target__'], how='left')
        results_test = results_test.rename(columns= {'__target__': 'TARGET'})

        i=0
        for real,pred in zip(Y_test,predictions):
            print(real,pred)
            i+=1
            if i>5:
                break

        print(f1_score(Y_test, predictions, average=None))
        print(classification_report(Y_test,predictions))
        print(confusion_matrix(Y_test, predictions, labels=[1,0]))