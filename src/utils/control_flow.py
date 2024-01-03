# -*- coding: utf-8 -*-


"""
Utilities for control flow
"""


class Register:
    """
    Register instances are usually used as decorators, which behaves like "factory pattern".
    It helps get rid of the tedious if-else clauses.
    """

    def __init__(self):
        self._register_map = dict()

    def get(self, name):
        return self._register_map.get(name)

    def build(self, name, *args, **kwargs):
        return self._register_map[name](*args, **kwargs)

    def __call__(self, name):
        def _register(func):
            if name in self._register_map:
                raise KeyError("{} has been registered".format(name))
            if func is not None:
                self._register_map[name] = func
            return func

        return _register
