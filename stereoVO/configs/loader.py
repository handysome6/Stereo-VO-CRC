from ruamel.yaml import YAML
from pathlib import Path

__all__ = ['yaml_parser', 'AttrDict']


class AttrDict(dict):
    """
    A dictionary that allows attribute-style access to its items.
    Compatible with Python 3.10+.
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self._convert_nested()

    def _convert_nested(self):
        """Recursively convert nested dicts to AttrDict."""
        for key, value in list(self.items()):
            if isinstance(value, dict) and not isinstance(value, AttrDict):
                super(AttrDict, self).__setitem__(key, AttrDict(value))
            elif isinstance(value, list):
                super(AttrDict, self).__setitem__(
                    key,
                    [AttrDict(item) if isinstance(item, dict) and not isinstance(item, AttrDict) else item
                     for item in value]
                )

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        if key.startswith('_'):
            super().__setattr__(key, value)
        else:
            self[key] = value

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{item}'")


def yaml_parser(path):
    """
    :param path(str):path of the yaml file relative to the root directory

    Returns:
        contents (AttrDict): contains the data loaded in memory
    """

    if isinstance(path, str):
        path = Path(path)

    with open(path, 'r', encoding='utf-8') as stream:
        contents = (YAML().load(stream))

    return AttrDict(contents)
    
