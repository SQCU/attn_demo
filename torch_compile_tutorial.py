#torch_compile_tutorial.py
import torch

def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b
opt_foo1 = torch.compile(foo)
print(opt_foo1(torch.randn(10, 10), torch.randn(10, 10)))

#C:\Program Files (x86)\Windows Kits\10\Include
#C:\Program Files (x86)\Windows Kits\10\Include\10.0.19041.0\ucrt
#...
