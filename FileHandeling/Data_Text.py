if __name__ == "__main__":
    #model = ImageBoundrybox.data_loader(os.path.join(option.PathToMathWriting, "train"))
    
    #else:
    #%%
    from FileHandeling.Lmdb import create_lmdb, read_entire_lmdb, read_from_lmdb
    from Options.OptionsUser import OptionsUser as option
    import os
    from glob import glob
    from tqdm import tqdm
    from FileHandeling.img2array import img2array
    from FileHandeling.imkml2img.inkml2img import inkml2img
    from FileHandeling.img2array import letterbox_bw
    import xml.etree.ElementTree as ET
    import numpy as np
    from skimage.transform import resize
    
    #if not os.path.exists(option.PathToMathWritingLmdbTrain):
    #    files = glob(os.path.join(option.PathToMathWriting, "train", "*.inkml"))
    #    print(files)
    #    data = {}
    #    for file in tqdm(files):
    #        new_file = file.replace(".inkml", ".png")
    #        inkml2img(file,new_file,size = (400,50))
    #        image = img2array(new_file, size = (400,50))
    #        os.remove(new_file)
    #
    #        with open(file) as f:
    #            label = f.read().split("\n")[5].removeprefix('<annotation type="normalizedLabel">').removesuffix(
    #            '</annotation>')
    #            #print(file.split("\\")[-1].replace(".inkml", ""), label)
    #            data[file.split("\\")[-1].replace(".inkml", "")] = (label, image)
    #
    #    create_lmdb(option.PathToMathWritingLmdbTrain, data)
    
    
    files = glob(os.path.join(option.PathToText, "*", "*.png"))
    files_xml = glob(os.path.join(option.PathToText, "*", "*.xml"))
    
    
    files_dict = {i.split("\\")[-1].split(".")[0]: i for i in tqdm(files)}
    files_xml_dict = {i.split("\\")[-1].split(".")[0]: i for i in tqdm(files_xml)}
    
    
    file_pair = [(files_dict[i], files_xml_dict[i]) for i in tqdm(files_dict)]
    
    data = {}
    
    for file, xml in tqdm(file_pair):
        array = img2array(file)
        
        tree = ET.parse(xml)
        root = tree.getroot()
        mc_text = root.findall(".//machine-print-line")
        lines = root.findall(".//handwritten-part/line")
        txt = ""
        for text in mc_text:
            txt += text.attrib.get("text")
            txt += " "
        
        
        asy = []
        dsy = []
        
        for line in lines:
            asy.append(int(line.attrib.get("asy")))
            dsy.append(int(line.attrib.get("dsy")))
        
        
        top, bottom = min(asy), max(dsy)
        
        array = array[top:bottom]
        
        array = letterbox_bw(array,desired_width = 400, desired_height = 400, pad_value = 1)
        data[file.split("\\")[-1].split(".")[0]] = (txt, array)
    
    print(data)
    create_lmdb(os.path.join(option.PathToTextLmdb, "All.lmdb"), data)
    
    
    
    print("Creating test, training, and validation data")
    
    data = read_entire_lmdb(os.path.join(option.PathToTextLmdb, "All.lmdb"))
    print(data)
    test_data = {}
    training_data = {}
    validation_data = {}
    
    for key in tqdm(data.keys()):
        r = random.random()
        if r < 0.8:
            training_data[key] = data[key]
        elif r < 0.9:
            validation_data[key] = data[key]
        else:
            test_data[key] = data[key]
    
    create_lmdb(os.path.join(option.PathToTextLmdb, "test.lmbd"), test_data)
    create_lmdb(os.path.join(option.PathToTextLmdb, "training.lmbd"), training_data)
    create_lmdb(os.path.join(option.PathToTextLmdb, "validation.lmbd"), validation_data)