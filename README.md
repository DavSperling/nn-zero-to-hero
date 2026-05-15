# 01 - Micrograd

Implémentation from scratch d'un mini autograd à la Karpathy, puis 
construction et entraînement d'un MLP sur un dataset jouet.

## Contenu
- `value.py` — autograd : add, mul, pow, tanh, exp + opérateurs (-, neg, sub, radd)
- `neuron.py` — classe Neuron (poids + biais + activation tanh)
- `layer.py` — classe Layer (n neurones en parallèle)
- `mlp.py` — classe MLP (chaîne de Layers)
- `training.py` — boucle d'entraînement sur dataset jouet
- `test.py` — 8 tests unitaires

## Résultats
- 8/8 tests passent
- MLP(3, [4, 4, 1]) entraîné sur 4 exemples
- Loss : 5.23 → 0.027 en 50 epochs (LR=0.05)

## Ce que j'ai appris
- Chain rule sur un DAG via tri topologique
- Forward / backward / gradient descent
- Architecture en couches d'un réseau de neurones
- Pattern PyTorch (parameters(), zero_grad, __call__)
- Pourquoi le zero_grad est essentiel entre itérations