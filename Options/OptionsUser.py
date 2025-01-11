from Options.Option import Option
import Options.ClassPropertyUtils as cpu



class OptionsUser(Option):
    
    
    
    @cpu.class_property
    def PathToMathWriting(cls) -> str:
        return r"C:\Users\user\Documents\GitHub\MathWriting"
    
    @cpu.class_property
    def PathToHMSD(cls) -> str:
        return r"A:\Overførsler\data\extracted_images"
    
    @cpu.class_property
    def PathToLetterCSV(cls) -> str:
        return r"A:\OneDrive - Danmarks Tekniske Universitet\letterImages.csv"
    
    @cpu.class_property
    def PathToLetterTrain(cls) -> str:
        return r"A:\OneDrive - Danmarks Tekniske Universitet\letterTraining.csv"
    
    @cpu.class_property
    def PathToLetterTest(cls) -> str:
        return r"A:\OneDrive - Danmarks Tekniske Universitet\letterTest.csv"
    
    @cpu.class_property
    def PathToLetterValidation(cls) -> str:
        return r"A:\OneDrive - Danmarks Tekniske Universitet\letterValidation.csv"
    


if __name__ == "__main__":
    print(OptionsUser.PathToMathWriting)
    print(OptionsUser.PathToHMSD)
    print(OptionsUser.PathToLetterCSV)
    print(OptionsUser.PathToLetterTrain)
    print(OptionsUser.PathToLetterTest)
    print(OptionsUser.PathToLetterValidation)
    