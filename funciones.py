from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

#Implementación de Minería de datos
def select_clusters(points, loops, init="k-means++", n_init=10, max_iter=300, random_state=42):
    """
        Recibe:
            points: Dataframe de pandas o array de numpy con los datos
            loops: cuantos cluster se experimentarán
            init, n_init, max_iter, random_state: argumentos de funcion kmeans
    """
    inertia_clusters = list() #lista donde se guardará la inercia de cada cluster

    for i in range(1, loops + 1):
        kmeans = KMeans(n_clusters=i, init=init, n_init=n_init, max_iter=max_iter, random_state=random_state, n_jobs=-1) # kmeans por cada numero de cluster

        kmeans.fit(points)
        
        inertia_clusters.append( kmeans.inertia_)

    return inertia_clusters

#Implementación de Minería de datos
def plot_results_codo(inertials, patron, Y="Inertia"):
    """
        Recibe:
            inertials: inertia en clusters k
        Regresa:
            None
    """
    x = range(1,len(inertials)+1) 
    plt.plot(x, inertials, patron, markersize=8, lw=2)
    plt.grid(True)
    plt.xlabel('Num Clusters')
    plt.ylabel(Y)