import shutil

import lmdb
import pickle
import cv2
import os

def create_lmdb(db_path, data):
    """
    Create an LMDB database and store key-value pairs provided in a dictionary.

    Parameters:
    - db_path: The path to create the LMDB database (e.g., './database.lmdb')
    - data: A dictionary containing key-value pairs to store in LMDB.
            Keys must be strings, and values can be any serializable object.
    """
    # If the LMDB path already exists, clear it to avoid conflicts
    if os.path.exists(db_path):
        shutil.rmtree(db_path)  # Remove the old LMDB directory if it exists
    
    # Dynamically calculate the required map_size based on data size and add some buffer
    total_size = sum(len(pickle.dumps(value, protocol = pickle.HIGHEST_PROTOCOL)) for value in data.values())
    map_size = int(total_size * 1.5)  # Add a 50% buffer to the required size
    
    # Create the LMDB environment with the calculated map_size
    env = lmdb.open(db_path, map_size = map_size)
    
    with env.begin(write = True) as txn:
        # Loop through the dictionary and insert key-value pairs
        for key, value in data.items():
            # Serialize the value
            serialized_value = pickle.dumps(value, protocol = pickle.HIGHEST_PROTOCOL)
            # Encode key as bytes (LMDB requires keys to be bytes)
            txn.put(key.encode('ascii'), serialized_value)
    
    env.close()
    print(f"Created LMDB at: {db_path}, with map_size: {map_size} bytes")


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

def read_entire_lmdb(db_path):
    """
    Reads and returns the entire content of the LMDB database.
    
    Parameters:
    - db_path: Path to the LMDB file (e.g., './my_dataset.lmdb')
    
    Returns:
    - A dictionary containing key-value pairs where:
        - Keys are string identifiers.
        - Values are the deserialized objects (e.g., images, labels, etc.).
    """
    data = {}  # Dictionary to store the database content
    env = lmdb.open(db_path, readonly = True, lock = False)  # Open the database in read-only mode
    with env.begin(write = False) as txn:  # Begin a read transaction
        cursor = txn.cursor()  # Use a cursor to iterate through the database
        for key, value in cursor:  # Iterate over all key-value pairs
            # Deserialize the value
            deserialized_value = pickle.loads(value)
            # Store the key-value pair in the dictionary (decode key to string for readability)
            data[key.decode('ascii')] = deserialized_value
    env.close()
    return data