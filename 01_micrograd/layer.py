from neuron import Neuron

class Layer:
    def __init__(self, nin, nout):
        """
        nin  : nombre d'entrées (= dimension des inputs)
        nout : nombre de neurones (= dimension de la sortie de la couche)
        """
        self.neurons = [ Neuron(nin) for _ in range(nout)]

    def __call__(self , x):
        """
        x : liste de Values (les entrées de la couche)
        Retourne : liste des sorties de chaque neurone
        """
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs


    def parameters(self):
        """
        Retourne TOUS les paramètres de la couche : poids + biais de chaque neurone.
        """
        params = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params
    