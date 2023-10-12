import json


def save_data(filename, data):
    """
    Save data to a JSON file.

    Args:
        filename (str): The name of the JSON file.
        data (dict): The data to be saved.

    Returns:
        None
    """
    with open(filename, "w") as json_file:
        json.dump(data, json_file, indent=4)


def load_data(filename):
    """
    Load data from a JSON file.

    Args:
        filename (str): The name of the JSON file.

    Returns:
        dict: The loaded data.
    """
    try:
        with open(filename, "r") as json_file:
            loaded_data = json.load(json_file)
        return loaded_data
    except FileNotFoundError:
        return None
