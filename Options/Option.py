from abc import ABC, abstractmethod
import Options.ClassPropertyUtils as cpu


class Option(ABC):
    @cpu.class_property
    @abstractmethod
    def PathToMathWriting(cls) -> str:
        pass
    
    
    @cpu.class_property
    @abstractmethod
    def PathToHMSD(cls) -> str:
        pass
    
    @cpu.class_property
    @abstractmethod
    def PathToLetterCSV(cls) -> str:
        pass
    
    @cpu.class_property
    @abstractmethod
    def PathToLetterTrain(cls) -> str:
        pass
    
    @cpu.class_property
    @abstractmethod
    def PathToLetterTest(cls) -> str:
        pass
    
    @cpu.class_property
    @abstractmethod
    def PathToLetterValidation(cls) -> str:
        pass
    
    @cpu.class_property
    @abstractmethod
    def PathToLetterNet(cls) -> str:
        pass
    
    @cpu.class_property
    @abstractmethod
    def PathToTextLmdb(cls) -> str:
        pass
    
    @cpu.class_property
    @abstractmethod
    def PathToText(cls) -> str:
        pass
    
    @cpu.class_property
    @abstractmethod
    def PathToLatexLmdb(cls) -> str:
        pass
