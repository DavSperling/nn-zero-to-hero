class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other):
        out = Value(
            data=self.data + other.data,
            _children=(self, other),
            _op='+'
        )

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(
            data=self.data * other.data,
            _children=(self, other),
            _op='*'
        )

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
    def __pow__(self, other):
        out = Value(
            data=self.data ** other,
            _children=(self,),
            _op='**'
        )
        def _backward():
            self.grad +=(other * self.data**(other-1) * out.grad)

        out._backward = _backward
        return out
    
    def tanh(self):
        import math
        t = math.tanh(self.data)
        out = Value(
            data= t ,
            _children=(self,),
            _op='tanh'
        )
        def _backward():
            self.grad += ( 1- t**2 )*out.grad 

        out._backward = _backward
        return out
    
    def exp(self):
        import math
        out = Value(
            data= math.exp(self.data) ,    
            _children= (self, ), 
            _op='exp'
        )
        def _backward():
            self.grad += out.data * out.grad 
        out._backward = _backward
        return out
    



    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

