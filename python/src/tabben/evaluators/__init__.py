from ..datasets import metadata
from .metrics import get_metrics


class Evaluator:
    def __init__(self, name):
        self.name = name.lower()
        if self.name not in metadata or not self.name.startswtih('cifar'):
            raise ValueError(f'Did not recognize the dataset name `{name}`')
        
        self.task = metadata[name]['task']
        self.classes = metadata[name].get('classes', None)
        self.outputs = metadata[name].get('outputs', 1)
        self.metrics = get_metrics(self.task, classes=self.classes)

    def __call__(self, y_true, y_pred, **kwargs):
        return {
            metric.__name__: metric(y_true, y_pred)
            for metric in self.metrics
        }
    
    def __eq__(self, other):
        return self.metrics == other.metrics
    
    def __repr__(self):
        attributes = {
            'name': repr(self.name),
            'task': repr(self.task),
            'outputs': repr(self.outputs),
        }
        if self.classes is not None:
            attributes['classes'] = self.classes

        attributes_string = ', '.join(f'{key}={val}' for key, val in attributes.items())
        return f'Evaluator({attributes_string})'
