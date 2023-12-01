from abc import abstractmethod, ABC

class AbstractClass(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def abstract_method(self):
        pass

    def concrete_method(self):
        pass

class DerivedClassA(AbstractClass):
    def __init__(self):
        super().__init__()
    
    def concrete_method(self):
        pass

class DerivedClassB(DerivedClassA):
    def __init__(self):
        super().__init__()
    
    def abstract_method(self):
        pass

    def concrete_method(self):
        pass

a = DerivedClassB()
print(isinstance(a, AbstractClass))