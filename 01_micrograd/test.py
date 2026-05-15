from value import Value
import random 
from neuron import Neuron
from layer import Layer 
from mlp import MLP 


def test_add_and_mul():
    """Test du calcul forward + backward sur d = a*b + c"""
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    d = a * b + c
    
    # Forward
    assert d.data == 4.0, f"Expected d.data=4.0, got {d.data}"
    
    # Backward
    d.backward()
    assert a.grad == -3.0, f"Expected a.grad=-3.0, got {a.grad}"
    assert b.grad == 2.0, f"Expected b.grad=2.0, got {b.grad}"
    assert c.grad == 1.0, f"Expected c.grad=1.0, got {c.grad}"
    
    print("✓ test_add_and_mul passed")
    print(f"  d.data = {d.data}")
    print(f"  a.grad = {a.grad}")
    print(f"  b.grad = {b.grad}")
    print(f"  c.grad = {c.grad}")

def test_pow():
    a = Value(3.0)
    b = a ** 2
    assert b.data == 9.0, f"Expected b.data=9.0, got {b.data}"
    b.backward()
    assert a.grad == 6.0, f"Expected a.grad=6.0, got {a.grad}"
    print("✓ test_pow passed")
    print(f"  b.data = {b.data}")
    print(f"  a.grad = {a.grad}")


def test_tanh():
    a = Value(0.5)
    b = a.tanh()
    b.backward()
    # tanh(0.5) ≈ 0.4621, et 1 - 0.4621² ≈ 0.7864
    assert abs(b.data - 0.4621) < 0.001, f"b.data = {b.data}"
    assert abs(a.grad - 0.7864) < 0.001, f"a.grad = {a.grad}"
    print("✓ test_tanh passed")
    print(f"  b.data = {b.data:.4f}")
    print(f"  a.grad = {a.grad:.4f}")


def test_exp():
    a = Value(1.0)
    b = a.exp()
    b.backward()
    # exp(1) ≈ 2.7183, dérivée = exp(1) ≈ 2.7183
    assert abs(b.data - 2.7183) < 0.001, f"b.data = {b.data}"
    assert abs(a.grad - 2.7183) < 0.001, f"a.grad = {a.grad}"
    print("✓ test_exp passed")
    print(f"  b.data = {b.data:.4f}")
    print(f"  a.grad = {a.grad:.4f}")

def test_neuron():
    random.seed(42)  # pour avoir des résultats reproductibles
    n = Neuron(3)    # un neurone à 3 entrées
    
    x = [Value(1.0), Value(-2.0), Value(3.0)]
    out = n(x)       # appelle __call__
    
    # Le résultat doit être une Value
    assert isinstance(out, Value), f"out devrait être une Value, pas {type(out)}"
    
    # Le résultat doit être entre -1 et 1 (tanh)
    assert -1 <= out.data <= 1, f"out.data = {out.data} hors [-1, 1]"
    
    # Backward doit marcher sans planter
    out.backward()
    
    # Tous les poids et le biais doivent avoir un gradient non-zero
    for wi in n.w:
        assert wi.grad != 0, f"Un poids a un gradient zéro"
    assert n.b.grad != 0, f"Le biais a un gradient zéro"
    
    # parameters() doit retourner 4 éléments (3 poids + 1 biais)
    assert len(n.parameters()) == 4
    
    print("✓ test_neuron passed")
    print(f"  out.data = {out.data:.4f}")
    print(f"  poids grads = {[round(wi.grad, 4) for wi in n.w]}")
    print(f"  biais grad = {n.b.grad:.4f}")

def test_layer():
    random.seed(42)
    layer = Layer(3, 4)    # 3 entrées, 4 neurones en sortie
    
    x = [Value(1.0), Value(-2.0), Value(3.0)]
    outs = layer(x)
    
    # La sortie doit être une LISTE de 4 Values
    assert isinstance(outs, list), f"outs devrait être une liste, pas {type(outs)}"
    assert len(outs) == 4, f"outs devrait avoir 4 éléments, a {len(outs)}"
    for o in outs:
        assert isinstance(o, Value), f"chaque sortie devrait être une Value"
        assert -1 <= o.data <= 1, f"sortie hors [-1, 1] (tanh)"
    
    # parameters() doit retourner 4 × (3 poids + 1 biais) = 16 paramètres
    params = layer.parameters()
    assert len(params) == 16, f"layer.parameters() devrait avoir 16 éléments, a {len(params)}"
    for p in params:
        assert isinstance(p, Value), f"chaque param devrait être une Value"
    
    print("✓ test_layer passed")
    print(f"  outs = {[round(o.data, 4) for o in outs]}")
    print(f"  nb params = {len(params)}")

def test_mlp():
    random.seed(42)
    # Réseau : 3 inputs → couche 4 → couche 4 → sortie 1
    mlp = MLP(3, [4, 4, 1])
    
    x = [Value(1.0), Value(-2.0), Value(3.0)]
    out = mlp(x)
    
    # La sortie doit être une SEULE Value (grâce au tweak de Layer.__call__)
    assert isinstance(out, Value), f"out devrait être une Value, pas {type(out)}"
    assert -1 <= out.data <= 1
    
    # Test backward
    out.backward()
    
    # parameters() doit retourner :
    # Layer 1 : 4 × (3 + 1) = 16
    # Layer 2 : 4 × (4 + 1) = 20
    # Layer 3 : 1 × (4 + 1) = 5
    # Total   : 41
    params = mlp.parameters()
    assert len(params) == 41, f"devrait avoir 41 params, a {len(params)}"
    
    # Vérifier que les params ont des gradients non-zéro après backward
    for p in params:
        assert p.grad != 0, "un param a un gradient zéro"
    
    print("✓ test_mlp passed")
    print(f"  out.data = {out.data:.4f}")
    print(f"  nb params = {len(params)}")
    print(f"  exemple de grad sur premier param: {params[0].grad:.4f}")
    
def test_operators():
    a = Value(3.0)
    
    # Soustraction Value - Value
    b = Value(1.0)
    c = a - b
    assert c.data == 2.0
    
    # Soustraction Value - nombre
    d = a - 1.0
    assert d.data == 2.0
    
    # Négation
    e = -a
    assert e.data == -3.0
    
    # Sum sur liste de Values (utilise __radd__)
    s = sum([Value(1.0), Value(2.0), Value(3.0)])
    assert s.data == 6.0
    
    # Multiplication Value * nombre
    f = a * 2
    assert f.data == 6.0
    
    print("✓ test_operators passed")


if __name__ == "__main__":
    test_add_and_mul()
    test_pow()
    test_tanh()
    test_exp()
    test_neuron()
    test_layer()
    test_mlp()
    test_operators()
    