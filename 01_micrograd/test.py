from value import Value


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


if __name__ == "__main__":
    test_add_and_mul()
    test_pow()
    test_tanh()
    test_exp()