import lmdb
import pickle
import cv2
import os

def create_lmdb(db_path, data_dir):
    """
    db_path: path to the LMDB file to create (e.g., './my_dataset.lmdb')
    data_dir: directory containing your images (JPEG, PNG, etc.)
    """
    # map_size sets the maximum size (in bytes) of the database
    # Make sure to pick something large enough for your dataset
    env = lmdb.open(db_path, map_size=1e11)  # 1e12 = 1 TB max size, adjust as needed
    
    with env.begin(write=True) as txn:
        # Loop through your image directory
        for i, filename in enumerate(sorted(os.listdir(data_dir))):
            # Skip non-image files if needed
            if not (filename.lower().endswith('.jpg') or filename.lower().endswith('.png')):
                continue
            
            path = os.path.join(data_dir, filename)
            # Read the image (using OpenCV here, but you could use PIL or another library)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue  # skip broken images
            
            # Serialize (pickle) the image array
            data = pickle.dumps(img, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Construct a key; keys must be bytes
            key = f"image-{i}".encode("ascii")
            # Put the (key, value) in the LMDB
            txn.put(key, data)
    
    env.close()
    print(f"Created LMDB at: {db_path}")

# Example usage:
# create_lmdb("./my_images.lmdb", "./images_folder")

def read_from_lmdb(db_path, key_str):
    """
    db_path: path to the LMDB file (e.g., './my_images.lmdb')
    key_str: string identifier used when creating data (e.g., 'image-0')
    """
    env = lmdb.open(db_path, readonly=True, lock=False)
    with env.begin(write=False) as txn:
        key = key_str.encode("ascii")
        data = txn.get(key)
        if data is None:
            print(f"No data found for key: {key_str}")
            return None
        # Deserialize the image
        img = pickle.loads(data)
    env.close()
    return img

# Example usage:
# loaded_img = read_from_lmdb("./my_images.lmdb", "image-0")
# if loaded_img is not None:
#     cv2.imshow("LMDB Image", loaded_img)
#     cv2.waitKey(0)
