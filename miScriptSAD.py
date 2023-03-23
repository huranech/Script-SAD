import imblearn
import sklearn
import pandas as pd
import pickle
import sys
import numpy as np
import getopt
import csv
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

#FUNCFUNCFUNCFUNCFUNCFUNCFUNCFUNCFUNCFUNCFUNCFUNCFUNC
#                   FUNCIONES                       F
#FUNCFUNCFUNCFUNCFUNCFUNCFUNCFUNCFUNCFUNCFUNCFUNCFUNC

# guardar modelo en el directorio de trabajo
def obt_features(file):
    datos = pd.read_csv(file)
    cabeceras = datos.columns.tolist()
    return cabeceras


def guardar_modelo(mejor_modelo):
    print("el mejor modelo tiene un f_score de: " + str(mejor_modelo[1]))
    nombre_modelo = "mejormodelo.sav"
    saved_model = pickle.dump(mejor_modelo[0], open(nombre_modelo, "wb"))
    print('se ha guardado el modelo')


# regenerar el modelo para aplicarlo a datos nuevos
def regenerar_modelo(nombre_modelo, fichero):
    X_nuevo = pd.read_csv(fichero)
    # se reescalan los valores de las features con una media de 0 y una desviación estándar de 1
    rescale_features = {'Largo de sepalo': 'AVGSTD', 'Ancho de sepalo': 'AVGSTD', 'Largo de petalo': 'AVGSTD', 'Ancho de petalo': 'AVGSTD'}
    for (feature_name, rescale_method) in rescale_features.items():
        if rescale_method == 'MINMAX':
            _min = X_nuevo[feature_name].min()
            _max = X_nuevo[feature_name].max()
            scale = _max - _min
            shift = _min
        else:
            shift = X_nuevo[feature_name].mean()
            scale = X_nuevo[feature_name].std()
        if scale == 0.:
            del X_nuevo[feature_name]
            print ('Feature %s was dropped because it has no variance' % feature_name)
        else:
            print ('Rescaled %s' % feature_name)
            X_nuevo[feature_name] = (X_nuevo[feature_name] - shift).astype(np.float64) / scale
    clf = pickle.load(open(nombre_modelo, 'rb'))
    resultado = clf.predict(X_nuevo)
    print(resultado)
    exit(0)


# crea un .csv con los datos de todos los experimentos
def csv_experimentos(datos_experimentos):
    with open("experimentos.csv", "w", newline="") as archivo:
        escritor = csv.writer(archivo)
        for fila in datos_experimentos:
            escritor.writerow(fila)


# función relacionada con la codificación en utf-8
def coerce_to_unicode(x):
    if sys.version_info < (3, 0):
        if isinstance(x, str):
            return unicode(x, 'utf-8')
        else:
            return unicode(x)
    else:
        return str(x)

#CODECODECODECODECODECODECODECODECODECODECODECODECODE
#                   CÓDIGO PRINCIPAL                C
#CODECODECODECODECODECODECODECODECODECODECODECODECODE

# ejecutar el código
if __name__ == '__main__':
    print('ARGV   :',sys.argv[1:])
    try:
        options,remainder = getopt.getopt(sys.argv[1:],'u:m:f:h:a:k:t:',['u=','model=','testFile=','h','a=','k=','t='])
    except getopt.GetoptError as err:
        print('ERROR:',err)
        sys.exit(1)
    print('OPTIONS   :',options)

    for opt,arg in options:
        if opt in ('-u','--ubicacion'):
            u = arg
        elif opt in ('-f', '--file'):
            f = arg
        elif opt in ('-m', '--model'):
            m = arg
        elif opt in ('-h','--help'):
            print("La forma correcta de usar el script es la siguiente: ")
            print("El script puede recibir 6 parámetros: -u, -m, -f, -a, -k, -t")
            print("-u: indica la ubicación en la que se hallan los archivos .csv y los modelos guardados.")
            print("-m: sólo se le debe asignar un valor si se pretende regenerar un modelo. Indica el nombre del archivo .sav en el que se halla el modelo.")
            print("-f: indica el nombre del archivo .csv tanto si se pretende entrenar el algoritmo como si se pretende realizar una predicción.")
            print("-a: sólo admite los valores knn y decisiontree. Representa el algoritmo que se desea usar para el entrenamiento.")
            print("-k: tódos los hiperparámetros se deben asignar aquí. Deben ir sin espacios y separados por comas.")
            print("el orden para los hiperparámetros del KNN son kmin,k,p,w donde kmin es el número mínimo de vecinos, k es el número máximo de vecinos,")
            print("p es el valor P máximo donde 1 representa la distancia de Manhatan y 2 es la distancia euclídea y, por último, w puede tomar los valores 'uniform' o 'distance'")
            print("la opción uniform otorga el mismo valor a los votos de los vecinos. La opción distance asigna un peso a los vecinos en función de su proximidad.")
            print("una ejemplo para usar este parametro en KNN sería: -k 1,5,2,uniform")
            print("si se estuviese usando el algoritmo decisiontree los hiperparámetros serían maxDepth,msx,msx_valor donde maxDepth es la profundidad máxima,")
            print("msx puede tomar los valores 'min_sample_split' ó 'min_sample_leaf' y msx_valor representa el valor que va a tomar la variable min_sample_X")
            print("una ejemplo para usar este parametro en decision trees sería: -k 5,min_sample_split,2")
            print("-t: indica el nombre del Target")
            print("nótese que si se pretende regenerar un modelo no se precisará introducir los parámetros -a, -k ni -t")
            print("un ejemplo de como se podría invocar al .py para entrenar un modelo sería: python miScriptSAD.py -u ./ -f iris.csv -a knn -k 1,5,2,distance -t Especie")
            print("un ejemplo de como se podría invocar al .py para predecir clases utilizando un modelo sería: python miScriptSAD.py -u ./ -f predecir.csv -m mejormodelo.sav")
            exit(1)
        elif opt in ('-a', '--algorithm'):
            if arg not in ["knn", "decisiontree"]:
                print("el argumento -a debe ser knn, decisiontree o ambos")
                exit(0)
            else:
                a = arg
        elif opt in ('-k','--hiperparameters'):  # recoge los hiperparámetros en un orden específico separados sólo por comas (sin espacios)
            hiperparametros = arg.split(",")
            if a == "knn":
                w = []
                kmin = int(hiperparametros[0])  # número mínimo de vecinos
                k = int(hiperparametros[1])  # número máximo de vecinos
                p = int(hiperparametros[2])  # número máximo del parámetro "p"
                w.append(hiperparametros[3])  # peso de los votos de los vecinos
            elif a == "decisiontree":
                valor_min_s = []
                max_d = int(hiperparametros[0])  # max_depth
                min_s = hiperparametros[1]  # "min_sample_split" ó "min_sample_leaf" ó "ambos"
                if hiperparametros[2] == "3":  # valor de min_sample_X (1, 2 ó 3; donde el 3 indica 1 y 2)
                    valor_min_s = [1, 2]
                else:
                    valor_min_s.append(int(hiperparametros[2]))
        elif opt in ('-t', '--target'):
            target = arg

    if u == './':
        iFile = u+ str(f)
        if 'm' in locals():
            modelo = u+str(m)
            regenerar_modelo(modelo, iFile)
    else:
        iFile = u+"/" + str(f)
        if 'm' in locals():
            modelo = u+"/"+str(m)
            regenerar_modelo(modelo, iFile)
        
    # abrir el fichero .csv y cargarlo en un dataframe de pandas
    ml_dataset = pd.read_csv(iFile)
    # recopilar los nombres de las features y la clase
    nombres_cabeceras = obt_features(iFile)
    # seleccionar únicamente los features que nos interesan (se pueden quitar algunas features) 
    ml_dataset = ml_dataset[nombres_cabeceras]

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
    
    # [HARDCODE] cambiar los valores del target_map
    target_map = {'Iris-versicolor': 0, 'Iris-virginica': 1, 'Iris-setosa': 2}
    print(target_map)
    ml_dataset['__target__'] = ml_dataset[target].map(str).map(target_map)
    del ml_dataset[target]

    # se eliminan las filas para las que el TARGET es null / se pasan los datos que fueran float a Integer
    ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]
    ml_dataset['__target__'] = ml_dataset['__target__'].astype(np.int64)

    # se separa la muestra en los conjuntos train y test donde el test representa el 20% y la proporción de targets es la misma para todo
    train, test = train_test_split(ml_dataset,test_size=0.2,random_state=42,stratify=ml_dataset[['__target__']])

    # [HARDCODE] se escoge la forma en la que se van a tratar los valores faltantes
    drop_rows_when_missing = []
    impute_when_missing = [{'feature': 'Largo de sepalo', 'impute_with': 'MEAN'}, {'feature': 'Ancho de sepalo', 'impute_with': 'MEAN'}, {'feature': 'Largo de petalo', 'impute_with': 'MEAN'}, {'feature': 'Ancho de petalo', 'impute_with': 'MEAN'}]

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
    rescale_features = {'Largo de sepalo': 'AVGSTD', 'Ancho de sepalo': 'AVGSTD', 'Largo de petalo': 'AVGSTD', 'Ancho de petalo': 'AVGSTD'}
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

#kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk
#                   ALGORITMO KNN                   k
#kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk

    if a == "knn":
        # antes de realizar los experimentos vamos a crear un array para guardarlos
        datos_experimentos = []
        cabecera = ["Experimento", "Precision", "Recall", "F_Score(mac/mic/avg/none)"]
        datos_experimentos.append(cabecera)

        # creamos una tupla para guardar el mejor modelo con su f_score
        mejor_modelo = (None, 0)

        # primero llenamos un array con todos los valores posibles de k
        barridoK = []
        for numero in range(kmin, k + 1):
            if numero == 0:
                pass # no ocurre nada, no se permite el valor 0
            elif not numero % 2 == 0:
                barridoK.append(numero)

        # conjunto de bucles donde sucede el barrido de hiperparámetros.
        for parametroK in barridoK:
            for parametroP in range(1, p + 1):
                if w[0] == "ambos":
                    w = ["uniform", "distance"]
                for parametroW in w:

                    # se crea el modelo con los hiperparámetros seleccionados
                    clf = KNeighborsClassifier(n_neighbors=parametroK,
                                        weights=parametroW,
                                        algorithm='auto',
                                        leaf_size=30,
                                        p=parametroP)
                    
                    # se establece el peso de cada clase
                    clf.class_weight = "balanced"  # none / balanced

                    # se imprimen los detalles sobre los hiperparámetros
                    print("experimento con " + "k = " + str(parametroK) + ", p = " + str(parametroP) + ", w = " + parametroW)

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
                    
                    datos_experimentos.append(["k=" + str(parametroK) + ",p=" + str(parametroP) + "," + str(w), str(precision_score(Y_test, predictions, average='micro')), str(recall_score(Y_test, predictions, average='micro')), str(f1_score(Y_test, predictions, average='micro'))])
                    print(f1_score(Y_test, predictions, average='micro'))
                    print(classification_report(Y_test,predictions))
                    print(confusion_matrix(Y_test, predictions, labels=[1,0]))

                    # guardamos el modelo en una variable siempre y cuando éste sea mejor que el anterior
                    if f1_score(Y_test, predictions, average='micro') > mejor_modelo[1]:
                        mejor_modelo = (clf, f1_score(Y_test, predictions, average='micro'))
        
        # creamos un archivo .csv con todos los experimentos.
        csv_experimentos(datos_experimentos)

        # se guarda el mejor modelo usando pickle
        guardar_modelo(mejor_modelo)

#ºººººººººººººººººººººººººººººººººººººººººººººººººººº
#                   DECISION TREE                   º
#ºººººººººººººººººººººººººººººººººººººººººººººººººººº
    elif a == "decisiontree":

        # antes de realizar los experimentos vamos a crear un array para guardarlos
        datos_experimentos = []
        cabecera = ["Experimento", "Precision", "Recall", "F_Score(mac/mic/avg/none)"]
        datos_experimentos.append(cabecera)

        # creamos una tupla para guardar el mejor modelo con su f_score
        mejor_modelo = (None, 0)

        # guardamos los max_depth impares entre dos números en un array
        barridoMaxD = []
        for numero in range(1, max_d + 1):
            if not numero % 2 == 0:
                barridoMaxD.append(numero)

        # conjunto de bucles propio del barrido de hiperparámetros
        for parametroMaxD in barridoMaxD:
            for parametroValorMinS in valor_min_s:
                if min_s == "min_samples_leaf":
                    clf = DecisionTreeClassifier(
                                        random_state = 1337,
                                        criterion = 'gini',
                                        splitter = 'best',
                                        max_depth = parametroMaxD,
                                        min_samples_leaf = parametroValorMinS)
                elif min_s == "min_samples_split":
                    clf = DecisionTreeClassifier(
                                        random_state = 1337,
                                        criterion = 'gini',
                                        splitter = 'best',
                                        max_depth = parametroMaxD,
                                        min_samples_split = parametroValorMinS)
                elif min_s == "ambos":  # no :(
                    clf = DecisionTreeClassifier(
                                        random_state = 1337,
                                        criterion = 'gini',
                                        splitter = 'best',
                                        max_depth = parametroMaxD,
                                        min_samples_split = parametroValorMinS)

                # se establece el peso de cada clase
                clf.class_weight = "balanced"  # none / balanced

                # se imprimen los detalles sobre los hiperparámetros
                print("experimento con " + "max_depth = " + str(parametroMaxD) + ", msx = " + min_s + ", msx_value = " + str(parametroValorMinS))

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
                
                datos_experimentos.append(["max_depth=" + str(parametroMaxD) + "," + str(min_s) + ",msx_value=" + str(parametroValorMinS), str(precision_score(Y_test, predictions, average='micro')), str(recall_score(Y_test, predictions, average='micro')), str(f1_score(Y_test, predictions, average='micro'))])
                print(f1_score(Y_test, predictions, average='micro'))
                print(classification_report(Y_test,predictions))
                print(confusion_matrix(Y_test, predictions, labels=[1,0]))

                # guardamos el modelo en una variable siempre y cuando éste sea mejor que el anterior
                if f1_score(Y_test, predictions, average='micro') > mejor_modelo[1]:
                    mejor_modelo = (clf, f1_score(Y_test, predictions, average='micro'))

        # creamos un archivo .csv con todos los experimentos.
        csv_experimentos(datos_experimentos)

        # se guarda el mejor modelo usando pickle
        guardar_modelo(mejor_modelo)