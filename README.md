# Perceptron
## Réseau de neurone mono-couche

Le perceptron de Rosemblatt introduit en 1957 définit un premier algorithme d’apprentissage sur un
réseau de neurones. Il a une seule couche et une sortie binaire afin de classifier des données dans deux
classes possibles. L’apprentissage se fait par une descente de gradient qui minimise une moyenne
des écarts à la frontière de décision. On démontre aussi que l’algorithme de Rosemblatt converge
vers une solution qui dépend des conditions initiales si les données d’entraînement sont séparables
linéairement, et ne converge pas si elles ne sont pas séparables.

Le perceptron peut être vu comme le type de réseau de neurones le plus simple. Ce type de réseau
neuronal ne contient aucun cycle (il s’agit d’un réseau de neurones à propagation avant).
