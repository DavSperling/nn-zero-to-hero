import random
from value import Value

class Neuron:
    def __init__(self, nin):
        """
        nin : nombre d'entrées de ce neurone
        On initialise nin poids + 1 biais, tous avec des valeurs random entre -1 et 1.
        Tous sont des Value (pour pouvoir calculer leurs gradients).
        """
        # TODO 1 : créer self.w comme une LISTE de nin Values
        # Chaque Value a comme data : random.uniform(-1, 1)
        # Indice : utilise une list comprehension : [Value(...) for _ in range(nin)]
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        
        # TODO 2 : créer self.b comme une seule Value
        # Avec data : random.uniform(-1, 1)
        self.b = Value(random.uniform(-1,1)) 
    
    def __call__(self, x):
        """
        x : liste de Values (les entrées), même taille que self.w
        Calcule : tanh(sum(wi * xi) + b)
        Retourne une Value.
        """
        # TODO 3 : 
        # - démarre avec act = self.b
        # - pour chaque paire (wi, xi) zip(self.w, x), fait act = act + wi * xi
        # - applique tanh à la fin
        # - retourne le résultat
        biais = self.b
        for wi, xi in zip(self.w, x):
            biais +=wi*xi 
        out = biais.tanh()   # TODO : appliquer tanh
        return out
    
    def parameters(self):
        """Retourne tous les paramètres entraînables (utile plus tard pour le training)"""
        return self.w + [self.b]