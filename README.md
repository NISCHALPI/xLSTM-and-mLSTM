# Extended Long Short-Term Memory (xLSTM) Module

This module provides implementations for mLSTM and sLSTM based on the concepts outlined in the referenced paper. The module contains two classes, sLSTMCell and mLSTMCell, which are defined in the xLSTM.py file. The sLSTMCell class mirrors the structure of a standard LSTM cell, akin to PyTorch's implementation. On the other hand, the mLSTMCell class is designed to operate as an mLSTM cell. Both classes offer a familiar interface resembling PyTorch's LSTM cell, featuring a forward method for conducting forward passes and an init_hidden method to initialize the hidden state. Furthermore, the mLSTM and sLSTM classes are constructed similarly to PyTorch's LSTM class. The mLSTM class integrates the mLSTMCell class as its cell component, while the sLSTM class incorporates the sLSTMCell class as its cell component.

**Paper Reference:** [xLSTM: Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517)


## Usage

### Importing the Module

```python
import torch
from xLSTM_module import sLSTMCell, mLSTMCell
```

### Initializing the sLSTMCell Model

```python
input_size = 10
hidden_size = 20
cell = sLSTMCell(input_size, hidden_size)
```

### Initializing the mLSTMCell Model

```python
input_size = 10
hidden_size = 20
cell = mLSTMCell(input_size, hidden_size)
```

### Forward Pass

Both `sLSTMCell` and `mLSTMCell` have a `forward` method that performs a forward pass.

```python
# Example usage for sLSTMCell
x = torch.randn(5, 10)  # input tensor
internal_state = cell.init_hidden(5)  # initialize hidden state tuple
output, new_internal_state = cell(x, internal_state)
```

### Initializing Hidden State

You can initialize the hidden state of the model using the `init_hidden` method.

```python
batch_size = 5
hidden_state = cell.init_hidden(batch_size)
```

## References

- [xLSTM: Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517)

