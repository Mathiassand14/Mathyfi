class class_property:
    def __init__(self, getter):
        self.getter = getter
    
    def __get__(self, instance, cls):
        return self.getter(cls)
    
    def setter(self, setter):
        self._setter = setter
        return self
    
    def __set__(self, instance, value):
        if hasattr(self, "_setter"):
            self._setter(type(instance), value)
        else:
            raise AttributeError("can't set attribute")
    
    def deleter(self, deleter):
        self._deleter = deleter
        return self
    
    def __delete__(self, instance):
        if hasattr(self, "_deleter"):
            self._deleter(type(instance))
        else:
            raise AttributeError("can't delete attribute")
