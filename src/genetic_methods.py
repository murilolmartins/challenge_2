import math
import random


def generate_population(population_size, max_volume_per_individual=10) -> list:
    """    Generates a random population of individuals, where each individual is represented
    by a random volume (integer) between 1 and max_volume_per_individual.
    Args:
        population_size (int): The number of individuals to generate.
        max_volume_per_individual (int): The maximum volume for each individual.
    Returns:
        list: A list of integers representing the volumes of individuals in the population.
    """
    population = []

    if population_size > 0 and max_volume_per_individual > 0:
        for _ in range(population_size):
            volume = random.randint(1, max_volume_per_individual + 1)
            population.append(volume)
    else:
        print("Population size and max volume must be greater than 0.")
    return population


def generate_max_number_of_travels(population, truck_volume):
    """
    Calculates the theoretical minimum number of travels (optimistic)
    and a safe upper bound for the number of travels (pessimistic)
    to guide genetic algorithm initialization.

    Args:
        population (list of int): A list of individual item volumes.
        truck_volume (int): The maximum volume capacity of the truck for a single trip.

    Returns:
        tuple: (optimistic_min_travels, pessimistic_max_travels_for_initial_assignment)
    """
    total_volume = sum(population)
    num_items = len(population)

    optimistic_min_travels = 0
    if total_volume > 0 and truck_volume > 0:
        optimistic_min_travels = math.ceil(total_volume / truck_volume)
    elif total_volume > 0 and truck_volume <= 0:  # Truck has no capacity, but items exist
        optimistic_min_travels = float('inf')  # Cannot transport

    # A safe upper bound for initial random assignment of trips is the number of items.
    # In the worst case, each item might need its own trip.
    # We add a small buffer for randomness in initial solutions.
    pessimistic_max_travels_for_initial_assignment = num_items  # Default worst-case

    # If the optimistic_min_travels is higher than num_items (e.g., if many small items sum up),
    # then use the optimistic as a base for the max range.
    # Add a buffer to allow GA to explore beyond the theoretical minimum.
    pessimistic_max_travels_for_initial_assignment = max(
        pessimistic_max_travels_for_initial_assignment,
        optimistic_min_travels  # Use the optimistic as a minimum for the range
    )

    # Always ensure at least 1 if there are items, or 0 if no items.
    if num_items > 0 and pessimistic_max_travels_for_initial_assignment == 0:
        pessimistic_max_travels_for_initial_assignment = 1

    print(
        f"Theoretical minimum travels (optimistic): {optimistic_min_travels}")
    print(
        f"Upper bound for initial trip assignments: {pessimistic_max_travels_for_initial_assignment}")

    return [optimistic_min_travels, pessimistic_max_travels_for_initial_assignment]


# --- Example Usage (from previous code for context) ---
# Assuming generate_initial_individuals and calculate_fitness are defined as before.
# (They are not included here for brevity, but would be the full code from previous responses)

def calculate_fitness(individual_plan, all_item_volumes, truck_capacity):
    """
    Calculates the fitness of an individual (loading plan).
    Fitness is the number of trips required, with a high penalty for invalid plans.
    Lower fitness value (fewer trips) is better.

    Returns:
        dict: A dictionary containing:
              'fitness' (float): The number of trips or float('inf') if invalid.
              'is_valid' (bool): True if the plan is valid, False otherwise.
              'trip_volumes' (dict): A dictionary mapping trip numbers to their total volumes.
    """
    trip_volumes = {}
    max_trip_number = 0
    is_valid = True  # Assume valid until a constraint is violated

    if not all_item_volumes and not individual_plan:  # Case: No items to transport
        return {
            'fitness': 0.0,
            'is_valid': True,
            'trip_volumes': {}
        }

    for item_index, assigned_trip in enumerate(individual_plan):
        # Basic validation for chromosome integrity
        if item_index >= len(all_item_volumes) or assigned_trip < 0:
            is_valid = False
            break  # No need to continue if chromosome is malformed

        item_volume = all_item_volumes[item_index]

        if assigned_trip not in trip_volumes:
            trip_volumes[assigned_trip] = 0

        trip_volumes[assigned_trip] += item_volume
        max_trip_number = max(max_trip_number, assigned_trip)

        # Constraint check: capacity exceeded
        if trip_volumes[assigned_trip] > truck_capacity:
            is_valid = False
            # We don't break here immediately if you want to see all trip_volumes
            # but for a pure validity check, breaking here is more efficient.
            # For this context (initial generation), we can break.
            break

    if is_valid:
        return {
            'fitness': float(max_trip_number + 1),
            'is_valid': True,
            'trip_volumes': trip_volumes
        }
    else:
        # If invalid, set fitness to infinity
        return {
            'fitness': float('inf'),
            'is_valid': False,
            'trip_volumes': trip_volumes  # Still return volumes for debugging invalid ones
        }


def generate_initial_individuals(population, individuals_number, truck_volume, travel_bounds):
    initial_individuals = []
    num_items = len(population)

    min_possible_trips = travel_bounds[0]
    max_possible_trips_for_initial_assignment = travel_bounds[1]

    if num_items > 0:
        min_possible_trips = max(1, min_possible_trips)
    else:
        min_possible_trips = 0

    if num_items > 0:
        max_possible_trips_for_initial_assignment = max(
            min_possible_trips, max_possible_trips_for_initial_assignment)
        if max_possible_trips_for_initial_assignment == 0:
            max_possible_trips_for_initial_assignment = 1
    else:
        max_possible_trips_for_initial_assignment = 0

    for i in range(individuals_number):
        attempts = 0
        while True:
            attempts += 1
            if attempts > 10000:  # Safety break to prevent infinite loops for impossible setups
                print(
                    f"ALERTA: Não foi possível gerar um indivíduo inicial válido após {attempts} tentativas. Verifique parâmetros GA ou volumes/capacidades.")
                return []  # Return empty list to signal failure

            current_individual = [0] * num_items

            if num_items == 0:
                # If no items, a plan of 0 trips is implicitly valid
                # print("DEBUG_INITIAL_IND: Nenhum item para empacotar. Gerando indivíduo vazio.")
                break

            # Ensure max_trip_val_to_assign is reasonable
            # If max_possible_trips_for_initial_assignment is 0, then max_trip_val_to_assign should also be 0
            num_trips_for_this_individual = random.randint(
                min_possible_trips, max_possible_trips_for_initial_assignment + 1)

            max_trip_val_to_assign = num_trips_for_this_individual - 1

            if max_trip_val_to_assign < 0:  # Ensure we don't try to assign to negative trips
                max_trip_val_to_assign = 0

            for item_idx in range(num_items):
                # Ensure random.randint receives valid range: low <= high
                # If max_trip_val_to_assign is 0, it means all items go to trip 0
                current_individual[item_idx] = random.randint(
                    0, max_trip_val_to_assign + 1)

            fitness_result = calculate_fitness(
                current_individual, population, truck_volume
            )

            # print(f"DEBUG_INITIAL_IND: Tentativa {attempts} para indivíduo {i}. Fitness: {fitness_result['fitness']}. Válido: {fitness_result['is_valid']}")

            if fitness_result['is_valid']:
                # print(f"DEBUG_INITIAL_IND: Indivíduo inicial válido gerado (fitness: {fitness_result['fitness']}): {current_individual}")
                break

        initial_individuals.append(current_individual)

    print(initial_individuals)

    return initial_individuals


def tournament_selection(population_with_fitness, k=3) -> list:
    """
    Realiza a seleção por torneio para escolher um único pai.

    Args:
        population_with_fitness (list of tuples): Uma lista onde cada item é
                                                 (individual, fitness_result_dict).
                                                 Ex: ( [0,1,0,...], {'fitness': 3.0, 'is_valid': True, ...} )
        k (int): O tamanho do torneio (número de indivíduos a competir).

    Returns:
        list: O indivíduo selecionado (o "cromossomo" que é a lista de atribuições de viagens).
    """
    if not population_with_fitness:
        return []  # Retorna vazio se a população estiver vazia

    # Seleciona k indivíduos aleatoriamente para o torneio
    tournament_competitors = random.sample(population_with_fitness, k)

    # Encontra o melhor indivíduo (menor fitness) entre os competidores
    # Inicializa com o primeiro competidor
    best_individual_tuple = tournament_competitors[0]

    for competitor_tuple in tournament_competitors:
        # Pega apenas o dicionário de resultado de fitness
        current_fitness_result = competitor_tuple[1]
        best_fitness_result = best_individual_tuple[1]

        # Compara o 'fitness' (número de viagens), lembrando que 'inf' é o pior
        if current_fitness_result['fitness'] < best_fitness_result['fitness']:
            best_individual_tuple = competitor_tuple

    # Retorna apenas o indivíduo (o cromossomo)
    return best_individual_tuple[0]


def two_point_crossover(parent1, parent2):
    """
    Realiza o cruzamento de dois pontos entre dois indivíduos (pais).

    Args:
        parent1 (list): O primeiro indivíduo (lista de atribuições de viagens).
        parent2 (list): O segundo indivíduo (lista de atribuições de viagens).

    Returns:
        tuple: Uma tupla contendo dois novos indivíduos (filhos) resultantes do cruzamento.
               Retorna (parent1, parent2) inalterados se o comprimento do cromossomo for muito pequeno.
    """
    length = len(parent1)

    # É necessário ter pelo menos 2 pontos de corte distintos, então o comprimento deve ser > 2.
    if length < 3:
        return parent1, parent2  # Não é possível cruzar, retorna os pais inalterados

    # Escolhe dois pontos de corte aleatórios
    # Os pontos de corte podem ser entre os genes, então os índices vão de 1 a length-1
    # random.sample(range(1, length), 2) garante 2 pontos distintos
    cut_point1, cut_point2 = sorted(random.sample(range(1, length), 2))

    # Cria os filhos
    child1 = [0] * length
    child2 = [0] * length

    # Filho 1:
    # Parte 1 (até cut_point1) do Pai 1
    child1[:cut_point1] = parent1[:cut_point1]
    # Parte 2 (entre cut_point1 e cut_point2) do Pai 2
    child1[cut_point1:cut_point2] = parent2[cut_point1:cut_point2]
    # Parte 3 (de cut_point2 em diante) do Pai 1
    child1[cut_point2:] = parent1[cut_point2:]

    # Filho 2: (partes invertidas)
    # Parte 1 (até cut_point1) do Pai 2
    child2[:cut_point1] = parent2[:cut_point1]
    # Parte 2 (entre cut_point1 e cut_point2) do Pai 1
    child2[cut_point1:cut_point2] = parent1[cut_point1:cut_point2]
    # Parte 3 (de cut_point2 em diante) do Pai 2
    child2[cut_point2:] = parent2[cut_point2:]

    return child1, child2


def mutate_individual(individual, mutation_rate, max_possible_trip_number):
    """
    Aplica mutação a um indivíduo. Cada gene (atribuição de item a viagem)
    tem uma chance de mutation_rate de ser alterado para um novo número de viagem aleatório.

    Args:
        individual (list): O indivíduo (cromossomo) a ser mutado.
        mutation_rate (float): A probabilidade (entre 0 e 1) de cada gene sofrer mutação.
        max_possible_trip_number (int): O número máximo de viagens que um gene mutado pode assumir (0-based).
                                        Geralmente, o upper bound da faixa de atribuição de viagens.

    Returns:
        list: O indivíduo mutado (pode ser uma cópia se nenhuma mutação ocorrer).
    """
    # Cria uma cópia mutável do indivíduo para não alterar o original diretamente
    mutated_individual = list(individual)

    # Se não houver itens, não há o que mutar
    if not mutated_individual:
        return mutated_individual

    # Garante que a faixa para mutação seja pelo menos 0 (para a viagem 0)
    # Se max_possible_trip_number é 0, significa que só existe a viagem 0.
    # Se for 5, pode ir de 0 a 5.
    max_trip_idx_for_mutation = max(
        0, max_possible_trip_number - 1)  # -1 para ser índice 0-based

    for i in range(len(mutated_individual)):
        if random.random() < mutation_rate:
            # Muta o gene: atribui um novo número de viagem aleatório
            # O novo número de viagem será entre 0 e max_trip_idx_for_mutation (inclusive)
            mutated_individual[i] = random.randint(
                0, max_trip_idx_for_mutation + 1)

    return mutated_individual
