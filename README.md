<h1 align = "center">:rocket: ANN :facepunch:</h1>

---

# Install
`pip install fast-ann`

# Usage
```python
from ann import ANN
import numpy as np

data = np.random.random((1000, 128)).astype('float32')

ann = ANN()
ann.train(data, index_factory='IVF4000, Flat', noramlize=True)

dis, idx = ann.search(data[:10])

print(dis)
print(idx)
```