from enum import Enum



class Options(Enum):
    def __init_subclass__(cls, **kwargs):
        
        options = ["PathToMathWriting", "PathToHASY"]
        # Check if OPTION_ONE is defined in the child class
        for option in options:
            if option not in cls.__members__:
                raise TypeError(f"{cls.__name__} must define an '{option}' member")
        super().__init_subclass__(**kwargs)