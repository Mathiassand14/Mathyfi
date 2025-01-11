from Options.Options import Options

class OptionsUser(Options):
    @property
    def PathToMathWriting(self) -> str:
        return r"C:\Users\user\Documents\GitHub\MathWriting"
    
    @property
    def PathToHMSD(self) -> str:
        return r"A:\Overførsler\data\extracted_images"
    