# Elements_Logiciels_ENSAE_Distance


Dans le cadre d'un projet de Machine Learning (ML)  sur lequel nous avons travaillé, nous devions estimer les prix d'un grand nombre de logements en France. Il  nous a paru pertinent de regarder si les prix étaient impactés par la distance entre le logement et les gares (SNCF et RATP) à proximité. Pour ce faire, il était nécessaire de calculer pour chaque logement la distance entre toutes les gares de France ce qui était très chronophage et donc allongeait considérablement le temps d'exécution de notre algorithme. C'est pourquoi nous avons décidé de mettre à profit les différents sujets abordés en cours et d'implémenter deux concepts de parallélisation, le multiprocessing et le multithreading. Notre projet est codé en Python et en Cython.


###  Organisation : 

* Dossier Graphiques : Sorties  Graphiques pour les différents algorithmes
* Dossier Résultats :  excels des sorties (distance de la gare la plus proche pour  chaques annonces)
* Elements_Logiciel_Distances.ipynb :  Notebook Introduction et comparaison vitesse/erreur entre geopandas, haversine et constitution de l'algorithme KdTree
* Multiprocess.py : Parallelisation Processeurs en Pyhton
* Multiprocess_KdTree.py :Parallelisation Processeurs en Pyhton  avec utilisation de KdTree
* Multithread.py : Parallelisation Threads en Python (ne fonctionne pas à cause du  GIL)
* multithreading cython.ipynb : Parallelisation Threads en Cython

Egalement:  données utilisées, compte-rendu et papier sur le KdTree
