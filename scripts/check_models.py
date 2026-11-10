from pathlib import Path
import ast,json
import torch
from torch import nn
import torch.nn.functional as F
root=Path(__file__).resolve().parents[1]
n=json.loads((root/'experiments.ipynb').read_text(encoding='utf-8'))
scope={'torch':torch,'nn':nn,'F':F}
for c in n['cells']:
    if c['cell_type']!='code': continue
    source=''.join(c['source'])
    tree=ast.parse(source)
    for node in tree.body:
        if isinstance(node,ast.ClassDef) and node.name in {'SimpleCNN','MLPAutoencoder','ConvAutoencoder'}:
            exec(compile(ast.Module(body=[node],type_ignores=[]),'<notebook-model>','exec'),scope)
for dropout,batchnorm in [(False,False),(True,False),(False,True)]:
    assert scope['SimpleCNN'](dropout,batchnorm)(torch.rand(4,1,28,28)).shape==(4,10)
for name,dims in [('MLPAutoencoder',[2,16,64]),('ConvAutoencoder',[16,64])]:
    for dim in dims:
        model=scope[name](dim);x=torch.rand(4,1,28,28);out=model(x)
        assert model.encode(x).shape==(4,dim)
        assert out.shape==x.shape
        assert torch.all((out>=0)&(out<=1))
        nn.MSELoss()(out,x).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
print('CNN and autoencoder shape, range and gradient checks passed; no training performed')
