from layer import Layer 

class MLP:
    def __init__(self, nin, nouts):
        """
        nin   : nombre d'entrées du réseau
        nouts : liste des tailles de sortie de chaque couche
                Ex: [4, 4, 1] = couche cachée 4 neurones, couche cachée 4 neurones, sortie 1 neurone
        """
        # Astuce : on construit la liste des tailles "frontières"
        # Ex : nin=3, nouts=[4, 4, 1]  →  sz = [3, 4, 4, 1]
        sz = [nin] + nouts
        
        # TODO : crée self.layers comme une liste de Layer
        # Chaque Layer i a comme entrée sz[i] et comme sortie sz[i+1]
        # Indice : list comprehension sur range(len(nouts))
        self.layers = [Layer( sz[i], sz[i + 1]) for i in range(len(nouts))]
    
    def __call__(self, x):
        """
        x : liste de Values (l'entrée du réseau)
        Forward pass à travers toutes les couches.
        """
        # TODO : pour chaque layer dans self.layers, passe x dedans et réassigne x
        # À la fin, retourne x
        # Pattern : boucle for simple
        for layer in self.layers:
            x =  layer(x)
        return x
    
    def parameters(self):
        """Retourne tous les paramètres de toutes les couches."""
        # Tu peux écrire ça en boucle simple, ou avec une double list comprehension :
        return [p for layer in self.layers for p in layer.parameters()]