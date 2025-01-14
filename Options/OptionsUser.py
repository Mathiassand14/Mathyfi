from Options.Option import Option
import Options.ClassPropertyUtils as cpu



class OptionsUser(Option):
    
    
    
    @cpu.class_property
    def PathToMathWriting(cls) -> str:
        return r"A:\Documents\mathwriting-2024"
    
    @cpu.class_property
    def PathToHMSD(cls) -> str:
        return r"A:\Overførsler\data\extracted_images"
    
    @cpu.class_property
    def PathToLetterCSV(cls) -> str:
        return r"A:\OneDrive - Danmarks Tekniske Universitet\Mathify\letterImages.csv"
    
    @cpu.class_property
    def PathToLetterTrain(cls) -> str:
        return r"A:\OneDrive - Danmarks Tekniske Universitet\Mathify\letterTraining.csv"
    
    @cpu.class_property
    def PathToLetterTest(cls) -> str:
        return r"A:\OneDrive - Danmarks Tekniske Universitet\Mathify\letterTest.csv"
    
    @cpu.class_property
    def PathToLetterValidation(cls) -> str:
        return r"A:\OneDrive - Danmarks Tekniske Universitet\Mathify\letterValidation.csv"
    
    @cpu.class_property
    def PathToLetterNet(cls) -> str:
        return r"A:\OneDrive - Danmarks Tekniske Universitet\Mathify\net"
    
    @cpu.class_property
    def PathToTextLmdb(cls) -> str:
        return r"A:\Documents\Mathyfi\TextLmdb"
    
    @cpu.class_property
    def PathToText(cls) -> str:
        return r"A:\OneDrive - Danmarks Tekniske Universitet\Mathify\Text"

if __name__ == "__main__":
    print(OptionsUser.PathToMathWriting)
    print(OptionsUser.PathToHMSD)
    print(OptionsUser.PathToLetterCSV)
    print(OptionsUser.PathToLetterTrain)
    print(OptionsUser.PathToLetterTest)
    print(OptionsUser.PathToLetterValidation)
    