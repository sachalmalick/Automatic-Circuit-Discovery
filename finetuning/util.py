import pickle

def load_pickled_data(filepath):
    with open(filepath, 'rb') as file:
        return pickle.load(file)
    
def pickle_obj(filepath, obj):
    with open(filepath, 'wb') as file:
        pickle.dump(obj, file)
