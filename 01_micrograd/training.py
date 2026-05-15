import random
from value import Value 
from mlp import MLP 

random.seed(42)

xs = [
    [Value(2.0), Value(3.0), Value(-1.0)],
    [Value(3.0), Value(-1.0), Value(0.5)],
    [Value(0.5), Value(1.0), Value(1.0)],
    [Value(1.0), Value(1.0), Value(-1.0)],
]

ys = [1.0, -1.0, -1.0, 1.0]

mlp = MLP( 3, [4 , 4 , 1])

# ───── TRAINING LOOP ─────

learning_rate = 0.05
n_epochs = 10

for epoch in range(n_epochs):
    # 1. FORWARD PASS
    ypreds = [mlp(x) for x in xs]
    
    # 2. LOSS (MSE)
    loss = sum((ypred - ytarget)**2 for ypred, ytarget in zip(ypreds, ys))
    
    # 3. ZERO GRAD : remet à zéro tous les gradients AVANT le backward
    for p in mlp.parameters():
        p.grad = 0.0
    
    # 4. BACKWARD : calcule les gradients de la loss par rapport aux paramètres
    loss.backward()
    
    # 5. UPDATE : descente de gradient
    for p in mlp.parameters():
        p.data -= learning_rate * p.grad
    
    # Affiche la progression toutes les 5 itérations
    if epoch % 5 == 0 or epoch == n_epochs - 1:
        print(f"Epoch {epoch:3d} : loss = {loss.data:.6f}")


# ──── PRÉDICTIONS FINALES ────

print("\n=== Après entraînement ===")
ypreds_final = [mlp(x) for x in xs]
for i, (yp, yt) in enumerate(zip(ypreds_final, ys)):
    print(f"Exemple {i}: prédit = {yp.data:+.4f}, attendu = {yt:+.1f}")


test = [Value(0.5), Value(1.0), Value(1.0)]
prediction = mlp(test)

print(f"Entrée     : {[v.data for v in test]}")
print(f"Prédiction : {prediction.data:+.4f}") 