+++
title = "Working with (Vanilla) PyTorch"
+++

This guide goes through how to use this package with the standard PyTorch workflow.

```python
!pip3 install tabben torch
```

## Loading the data

For this example, we'll use the poker hand prediction dataset.

```python
from tabben.datasets import OpenTabularDataset

ds = OpenTabularDataset('./data/', 'poker')
```

And let's just look at the input and output attributes:

```python
print('Input Attribute Names:')
print(ds.input_attributes)

print('Output Attribute Names:')
print(ds.output_attributes)
```

Since we're working with PyTorch, the `OpenTabularDataset` object above is a PyTorch `Dataset` object that we can directly feed into a `DataLoader`.

```python
from torch.utils.data import DataLoader

dl = DataLoader(ds, batch_size=8)

example_batch = next(iter(dl))
print(f'Input Shape (Batched): {example_batch[0].shape}')
print(f'Output Shape (Batched): {example_batch[1].shape}')
```

## Setting up a basic model

First, we'll create a basic model in PyTorch, just for illustration (you can replace this with whatever model you're trying to train/evaluate). It'll just be a feedforward neural network with a couple dense/linear layers (this probably won't perform well).

```python
from torch import nn
import torch.nn.functional as F


class ShallowClassificationNetwork(nn.Module):
    
    def __init__(self, num_inputs, num_classes):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, num_classes)
    
    def forward(self, inputs):
        # [b, num_inputs] -> [b, 32]
        x = F.relu(self.linear1(inputs))
        
        # [b, 32] -> [b, 32]
        x = F.relu(self.linear2(x))
        
        # [b, 32] -> [b, num_classes] (log(softmax(.)) computed over each row)
        x = F.log_softmax(self.linear3(x), dim=1)
        
        return x
    
    def predict(self, inputs):
        x = F.relu(self.linear1(inputs))
        x = F.relu(self.linear2(x))
        return F.softmax(self.linear3(x), dim=1)

```

```python
import torch

device = torch.device('cpu')  # change this to 'cuda' if you have access to a CUDA GPU
model = ShallowClassificationNetwork(ds.num_inputs, ds.num_classes).to(device)
```

## Training the model

Now that we have a basic model and a training dataset (the default split for `OpenTabularDataset` is the train split), we can train our simple network using a PyTorch training loop.

```python
from torch import optim

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

```

```python
from tqdm import trange

model.train()

training_progress = trange(30, desc='Train epoch')
for epoch in training_progress:
    running_loss = 0.
    running_acc = 0.
    running_count = 0
    
    for batch_input, batch_output in dl:
        optimizer.zero_grad()
        
        outputs = model(batch_input.float().to(device))
        preds = outputs.argmax(dim=1)
        
        loss = criterion(outputs, batch_output)
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
        running_acc += (preds == batch_output).sum()
        running_count += batch_input.size(0)
    
    training_progress.set_postfix({
        'train loss': f'{running_loss / running_count:.4f}',
        'train acc': f'{100 * running_acc / running_count:.2f}%',
    })

```

You can play around with the hyperparameters, but the model isn't likely to get particularly good performance like this. But, we'll go ahead and evaluate the final model (we ignored having a validation set in this guide) on the test set.

## Evaluating on the test set

```python
test_ds = OpenTabularDataset('./data/', 'poker', split='test')
test_dl = DataLoader(test_ds, batch_size=16)
print(len(test_ds))
```

Let's run the model and save its outputs for later evaluation (since we left the softmax operation for the loss when we defined the model above, we'll need to softmax the outputs to get probabilities).

```python
model.eval()

pred_outputs = []
gt_outputs = []

for test_inputs, test_outputs in test_dl:
    batch_outputs = model(test_inputs.float().to(device))
    pred_outputs.append(batch_outputs.detach().cpu())
    gt_outputs.append(test_outputs)
```

```python
test_pred_outputs = torch.softmax(torch.vstack(pred_outputs).detach().cpu(), axis=1)
test_gt_outputs = torch.hstack(gt_outputs).detach().cpu()
```

We can get the standard set of metrics and then evaluate the outputs of the test set on them.

```python
from tabben.evaluators import get_metrics

eval_metrics = get_metrics(ds.task, classes=ds.num_classes)
for metric in eval_metrics:
    print(f'{metric}: {metric(test_gt_outputs, test_pred_outputs)}')
```

---

This code was last run with the following versions (if you're looking at the no-output webpage, see the notebook in the repository for versions):

```python
from importlib.metadata import version

packages = ['torch', 'tabben']

for pkg in packages:
    print(f'{pkg}: {version(pkg)}')
```

```python

```
