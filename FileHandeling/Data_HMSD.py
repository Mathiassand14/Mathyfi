import os
from typing import List, Tuple

from Options.OptionsUser import OptionsUser


if __name__ == "__main__":
    dir: str = OptionsUser.PathToHMSD.value
    folders:List[Tuple[str,str]] = [(os.path.join(dir,d),d) for d in os.listdir(dir)]
    print(folders)
    #dict{i:i for i in folders}