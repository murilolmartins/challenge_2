import math
import random


def generate_population(population_size, max_volume_per_individual=10) -> list:
    """    Generates a random population of individuals, where each individual is represented
    by a random volume (integer) between 1 and max_volume_per_individual.
    Args:
        max_volume_general: Volume total da população
        population_size (int): The number of individuals to generate.
        max_volume_per_individual (int): The maximum volume for each individual.
    Returns:
        list: A list of integers representing the volumes of individuals in the population.
    """
    population = []
    max_volume_general = population_size * max_volume_per_individual
    if population_size > 0 and max_volume_per_individual > 0:
        total_volume_generated = 0
        for _ in range(population_size):
            if total_volume_generated > max_volume_general:
                break
            volume = random.randint(1, max_volume_per_individual + 1)
            total_volume_generated += volume
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
    Calcula o fitness de um indivíduo (plano de carregamento).
    Considera múltiplos fatores:
    1. Número de viagens
    2. Utilização do caminhão (quanto mais próximo da capacidade, melhor)
    3. Penalizações graduais para violações

    Retorna:
        dict: Um dicionário contendo:
              'fitness' (float): O valor de fitness (menor é melhor)
              'is_valid' (bool): True se o plano é válido
              'trip_volumes' (dict): Dicionário mapeando número da viagem para volume total
    """
    trip_volumes = {}
    max_trip_number = 0
    is_valid = True
    total_penalty = 0

    if not all_item_volumes and not individual_plan:
        return {
            'fitness': 0.0,
            'is_valid': True,
            'trip_volumes': {}
        }

    # Mapear volumes por viagem
    for item_index, assigned_trip in enumerate(individual_plan):
        # Validar tipo e valor de assigned_trip
        if not isinstance(assigned_trip, (int, float)) or isinstance(assigned_trip, list):
            return {
                'fitness': float('inf'),
                'is_valid': False,
                'trip_volumes': trip_volumes,
                'error': f'Trip assignment deve ser um número, não {type(assigned_trip)}'
            }

        if item_index >= len(all_item_volumes) or assigned_trip < 0:
            return {
                'fitness': float('inf'),
                'is_valid': False,
                'trip_volumes': trip_volumes,
                'error': 'Índice de item inválido ou viagem negativa'
            }

        item_volume = all_item_volumes[item_index]

        if assigned_trip not in trip_volumes:
            trip_volumes[assigned_trip] = 0

        trip_volumes[assigned_trip] += item_volume
        max_trip_number = max(max_trip_number, assigned_trip)

        # Penalização gradual para excesso de capacidade
        if trip_volumes[assigned_trip] > truck_capacity:
            excess = trip_volumes[assigned_trip] - truck_capacity
            # Penalidade proporcional ao excesso
            penalty = (excess / truck_capacity) * 2
            total_penalty += penalty
            is_valid = False

    # Número base de viagens
    num_trips = max_trip_number + 1
    base_fitness = float(num_trips)

    # Calcular utilização média e desvio padrão dos caminhões
    if trip_volumes:
        volumes = list(trip_volumes.values())
        avg_utilization = sum(volumes) / len(volumes)

        # Penalizar utilização muito baixa dos caminhões
        utilization_penalty = 0
        for volume in volumes:
            if volume < truck_capacity * 0.5:  # Menos de 50% utilizado
                utilization_penalty += 0.5
            elif volume < truck_capacity * 0.7:  # Menos de 70% utilizado
                utilization_penalty += 0.2

        # Penalizar distribuição desigual entre caminhões
        squared_diff_sum = sum((v - avg_utilization) ** 2 for v in volumes)
        std_dev = (squared_diff_sum / len(volumes)) ** 0.5
        distribution_penalty = std_dev / truck_capacity

        total_penalty += utilization_penalty + distribution_penalty

    final_fitness = base_fitness + total_penalty

    return {
        'fitness': final_fitness if is_valid else float('inf'),
        'is_valid': is_valid,
        'trip_volumes': trip_volumes
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
        k (int): O tamanho do torneio (número de indivíduos a competir).

    Returns:
        list: O indivíduo selecionado (o "cromossomo" que é a lista de atribuições de viagens).
    """
    if not population_with_fitness:
        return []

    # Aumenta a pressão seletiva ordenando primeiro
    sorted_population = sorted(
        population_with_fitness, key=lambda x: x[1]['fitness'])

    # Seleciona k indivíduos com preferência para os melhores
    tournament_size = min(k, len(sorted_population))
    # Dá mais chance para os melhores indivíduos serem selecionados
    weights = [1/(i+1) for i in range(len(sorted_population))]
    tournament_competitors = random.choices(
        sorted_population, weights=weights, k=tournament_size)

    # Retorna o melhor do torneio
    best_individual = min(tournament_competitors,
                          key=lambda x: x[1]['fitness'])

    return best_individual[0]


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
    Aplica mutação a um indivíduo de forma mais suave, com pequenas alterações.
    """
    # Garantir que individual é uma lista e não um dicionário
    if isinstance(individual, dict):
        return list(individual.values())

    mutated_individual = list(individual)

    if not mutated_individual:
        return mutated_individual

    max_trip_idx_for_mutation = max(0, max_possible_trip_number - 1)

    for i in range(len(mutated_individual)):
        if random.random() < mutation_rate:
            # Garantir que o valor atual é um número
            try:
                current_trip = int(mutated_individual[i])
            except (ValueError, TypeError):
                # Se não for possível converter para número, usa 0
                current_trip = 0

            # Mutação mais suave: 50% de chance de incrementar/decrementar
            # e 50% de chance de atribuir um valor aleatório
            if random.random() < 0.5:
                # Mutação suave: incrementa ou decrementa em 1
                delta = random.choice([-1, 1])
                new_trip = max(
                    0, min(max_trip_idx_for_mutation, current_trip + delta))
                mutated_individual[i] = new_trip
            else:
                # Mutação aleatória: mas com preferência por valores próximos
                variation = random.randint(1, 3)  # Varia em até 3 viagens
                if random.random() < 0.5:
                    variation = -variation
                new_trip = max(0, min(max_trip_idx_for_mutation,
                               current_trip + variation))
                mutated_individual[i] = new_trip

    return mutated_individual


def is_population_stagnated(fitness_history, stagnation_window=50, improvement_threshold=0.001):
    """
    Verifica se a população está estagnada comparando o progresso das últimas gerações.
    """
    if len(fitness_history) < stagnation_window:
        return False

    recent_fitness = fitness_history[-stagnation_window:]
    initial_fitness = recent_fitness[0]
    final_fitness = recent_fitness[-1]

    # Se não houve melhoria significativa
    improvement = (initial_fitness - final_fitness) / initial_fitness
    return improvement < improvement_threshold


def diversify_population(population, item_volumes, truck_capacity, mutation_rate, max_possible_trip_number):
    """
    Diversifica a população quando há estagnação.
    Mantém os 10% melhores e recria o resto com foco em soluções diferentes.
    """
    population_size = len(population)
    elite_size = max(1, population_size // 10)  # Mantém 10% dos melhores

    # Garantir que todos os indivíduos são listas válidas
    valid_population = []
    for ind in population:
        if isinstance(ind, tuple):  # Se for tupla (indivíduo, fitness)
            ind = ind[0]
        if isinstance(ind, dict):  # Se for dicionário
            ind = list(ind.values())
        if isinstance(ind, list):
            valid_population.append(ind)

    if not valid_population:
        # Se não temos indivíduos válidos, criar nova população do zero
        return generate_initial_individuals(item_volumes, population_size, truck_capacity, [0, max_possible_trip_number])

    # Avalia e ordena a população atual
    evaluated_pop = []
    for ind in valid_population:
        fitness_result = calculate_fitness(ind, item_volumes, truck_capacity)
        if fitness_result['is_valid']:
            evaluated_pop.append((ind, fitness_result))

    if not evaluated_pop:
        # Se não temos indivíduos válidos após avaliação
        return generate_initial_individuals(item_volumes, population_size, truck_capacity, [0, max_possible_trip_number])

    # Ordena por fitness
    evaluated_pop.sort(key=lambda x: x[1]['fitness'])

    # Mantém a elite
    new_population = [ind for ind, _ in evaluated_pop[:elite_size]]

    # Lista de estratégias de diversificação
    diversification_strategies = [
        lambda x: mutate_individual(
            # Mutação agressiva
            x, min(1.0, mutation_rate * 3), max_possible_trip_number),
        # Embaralha segmentos do indivíduo
        lambda x: random_shuffle_segments(x),
        lambda x: invert_segments(x),  # Inverte segmentos do indivíduo
    ]

    # Contador para evitar loop infinito
    max_attempts = population_size * 10
    attempts = 0

    # Para o resto da população, cria novos indivíduos com diferentes estratégias
    while len(new_population) < population_size and attempts < max_attempts:
        attempts += 1

        # Escolhe um indivíduo base da elite
        base_individual = list(random.choice(new_population))

        # Escolhe uma estratégia de diversificação aleatória
        strategy = random.choice(diversification_strategies)
        mutated = strategy(base_individual)

        # Verifica se o indivíduo gerado é válido
        if not isinstance(mutated, list):
            continue

        fitness_result = calculate_fitness(
            mutated, item_volumes, truck_capacity)
        if fitness_result['is_valid']:
            new_population.append(mutated)

    # Se não conseguimos gerar população suficiente, completa com indivíduos aleatórios
    while len(new_population) < population_size:
        new_ind = generate_initial_individuals(item_volumes, 1, truck_capacity, [
                                               0, max_possible_trip_number])
        if new_ind:
            new_population.append(new_ind[0])

    return new_population


def random_shuffle_segments(individual):
    """
    Embaralha segmentos do indivíduo para criar variação
    """
    if len(individual) < 2:
        return individual

    result = list(individual)
    segment_size = max(2, len(result) // 4)

    # Divide em segmentos e embaralha
    segments = [result[i:i + segment_size]
                for i in range(0, len(result), segment_size)]
    random.shuffle(segments)

    # Reconstrói o indivíduo
    result = []
    for segment in segments:
        result.extend(segment)

    return result


def invert_segments(individual):
    """
    Inverte segmentos aleatórios do indivíduo
    """
    if len(individual) < 2:
        return individual

    result = list(individual)
    segment_size = max(2, len(result) // 4)

    # Escolhe um segmento aleatório para inverter
    start = random.randint(0, len(result) - segment_size)
    end = start + segment_size

    # Inverte o segmento
    result[start:end] = reversed(result[start:end])

    return result


def check_and_merge_trips(individual, item_volumes, truck_capacity):
    """
    Tenta melhorar uma solução verificando se é possível combinar viagens.
    """
    if not individual:
        return individual

    # Garantir que individual é uma lista
    if isinstance(individual, dict):
        individual = list(individual.values())

    # Mapeia os volumes por viagem
    trip_volumes = {}
    for item_idx, trip in enumerate(individual):
        # Garantir que trip é um número
        try:
            trip_number = int(trip) if isinstance(
                trip, (int, float, str)) else 0
            if isinstance(trip, list):
                trip_number = 0  # valor padrão para listas
        except (ValueError, TypeError):
            trip_number = 0  # valor padrão para conversões inválidas

        if trip_number not in trip_volumes:
            trip_volumes[trip_number] = 0
        trip_volumes[trip_number] += item_volumes[item_idx]

    # Tenta combinar viagens parcialmente preenchidas
    trips = sorted(trip_volumes.keys())
    improved = True
    while improved:
        improved = False
        for i in range(len(trips)):
            for j in range(i + 1, len(trips)):
                trip1, trip2 = trips[i], trips[j]
                # Se as duas viagens juntas não excedem a capacidade
                if trip_volumes[trip1] + trip_volumes[trip2] <= truck_capacity:
                    # Combina as viagens
                    for idx in range(len(individual)):
                        # Garantir que estamos comparando números
                        current_trip = individual[idx]
                        if isinstance(current_trip, list):
                            current_trip = 0
                        try:
                            current_trip = int(current_trip)
                        except (ValueError, TypeError):
                            current_trip = 0

                        if current_trip == trip2:
                            individual[idx] = trip1
                    trip_volumes[trip1] += trip_volumes[trip2]
                    del trip_volumes[trip2]
                    trips.remove(trip2)
                    improved = True
                    break
            if improved:
                break

    return individual
