import random
import json
import os
from genetic_methods import generate_population


def save_population_to_file(population: list, filename="fixed_population.json"):
    """
    Saves the generated population to a JSON file.
    Args:
        population (list): The list of integers representing the population volumes.
        filename (str): The name of the file to save the population to.
    """
    try:
        with open(filename, 'w') as f:
            json.dump(population, f, indent=4)
        print(f"Population successfully saved to '{filename}'")
    except IOError as e:
        print(f"Error saving population to file: {e}")


def load_population_from_file(filename="fixed_population.json") -> list:
    """
    Loads a population from a JSON file.
    Args:
        filename (str): The name of the file to load the population from.
    Returns:
        list: The loaded list of integers representing the population volumes.
    """
    population = []
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                population = json.load(f)
            print(f"Population successfully loaded from '{filename}'")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from file: {e}")
        except IOError as e:
            print(f"Error loading population from file: {e}")
    else:
        print(f"File '{filename}' not found.")
    return population


if __name__ == "__main__":

    population = generate_population(
        population_size=200, max_volume_per_individual=50)
    save_population_to_file(population)
