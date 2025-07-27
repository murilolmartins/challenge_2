import random
import pygame
import sys
from genetic_methods import (
    generate_population,
    generate_max_number_of_travels,
    calculate_fitness,
    generate_initial_individuals,
    tournament_selection,
    two_point_crossover,
    mutate_individual,
    check_and_merge_trips,
    is_population_stagnated,
    diversify_population
)


def run_genetic_algorithm(
    num_items,
    max_item_volume,
    max_total_volume,
    truck_capacity,
    population_size,
    num_generations,
    crossover_rate,
    mutation_rate,
    tournament_size=3,
    num_elite_individuals=1  # Novo parâmetro para elitismo
):
    """
    Orchestrates the Genetic Algorithm with Pygame visualization, including elitism.

    Args:
        num_items (int): Total number of items to pack.
        max_item_volume (int): Maximum possible volume for a single item.
        truck_capacity (int): Capacity of the truck per trip.
        population_size (int): Number of individuals in each generation.
        num_generations (int): Total generations to run the GA.
        crossover_rate (float): Probability of crossover (0.0 to 1.0).
        mutation_rate (float): Probability of a gene mutating (0.0 to 1.0).
        tournament_size (int): Number of individuals in each selection tournament.
        num_elite_individuals (int): Number of best individuals to carry directly to the next generation.
    """
    # Inicialização do Pygame
    pygame.init()
    SCREEN_WIDTH = 1200
    SCREEN_HEIGHT = 800
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(
        "Algoritmo Genético para Empacotamento de Caminhões")

    # Cores
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GREEN = (0, 200, 0)
    RED = (200, 0, 0)
    BLUE = (0, 0, 200)
    GREY = (200, 200, 200)
    TRUCK_COLOR = (100, 100, 255)
    # ITEM_COLORS já não é usada na visualização atual (barra de preenchimento)

    # Fontes
    font = pygame.font.Font(None, 24)
    big_font = pygame.font.Font(None, 36)

    # Configuração do GA
    item_population_volumes = generate_population(num_items, max_item_volume, max_total_volume)
    travel_bounds = generate_max_number_of_travels(
        item_population_volumes, truck_capacity)
    max_possible_trip_for_mutation = travel_bounds[1]

    current_population = generate_initial_individuals(
        item_population_volumes, population_size, truck_capacity, travel_bounds
    )

    best_overall_solution = None
    best_overall_fitness = float('inf')
    best_overall_trip_volumes = {}
    current_generation = 0

    # Adiciona histórico de fitness para detectar estagnação
    fitness_history = []
    generations_without_improvement = 0
    best_fitness_ever = float('inf')

    # Loop Principal do Pygame e GA
    running = True
    ga_step_counter = 0
    GA_STEPS_PER_FRAME = 1

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Lógica do Algoritmo Genético (Executado por GA_STEPS_PER_FRAME) ---
        if ga_step_counter < num_generations:
            # 1. Avaliar a população atual
            evaluated_population = []
            for individual in current_population:
                # Tenta melhorar o indivíduo antes de avaliar
                optimized_individual = check_and_merge_trips(
                    list(individual), item_population_volumes, truck_capacity)
                fitness_result = calculate_fitness(
                    optimized_individual, item_population_volumes, truck_capacity)
                evaluated_population.append(
                    (optimized_individual, fitness_result))

            # Ordenar por fitness
            evaluated_population.sort(key=lambda x: x[1]['fitness'])

            # Atualizar histórico de fitness
            current_best_fitness = evaluated_population[0][1]['fitness']
            fitness_history.append(current_best_fitness)

            # Verificar melhoria
            if current_best_fitness < best_fitness_ever:
                best_fitness_ever = current_best_fitness
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            # Atualizar melhor solução global
            current_best_individual, current_best_fitness_result = evaluated_population[0]
            if current_best_fitness_result['fitness'] < best_overall_fitness:
                best_overall_fitness = current_best_fitness_result['fitness']
                best_overall_solution = current_best_individual
                best_overall_trip_volumes = current_best_fitness_result['trip_volumes']

            # Verificar estagnação
            if is_population_stagnated(fitness_history) or generations_without_improvement >= 100:
                print(
                    f"Estagnação detectada na geração {current_generation}. Diversificando população...")
                # Extrair apenas os indivíduos (sem o fitness) da população avaliada
                individuals_only = [ind for ind, _ in evaluated_population]
                current_population = diversify_population(
                    individuals_only,  # Passa apenas os indivíduos, não as tuplas
                    item_population_volumes,
                    truck_capacity,
                    mutation_rate,
                    max_possible_trip_for_mutation
                )
                generations_without_improvement = 0
                fitness_history = []  # Reinicia o histórico de fitness
                continue

            # 2. Preparar próxima geração
            next_generation_population = []

            # --- Elitismo ---
            # Copia os 'num_elite_individuals' melhores diretamente para a próxima geração
            for i in range(min(num_elite_individuals, len(evaluated_population))):
                # Pega apenas o indivíduo (cromossomo)
                next_generation_population.append(evaluated_population[i][0])

            # Preencher o resto da população
            # A seleção e cruzamento/mutação gerarão 'population_size - num_elite_individuals' novos indivíduos
            while len(next_generation_population) < population_size:
                # 3. Seleção
                parent1 = tournament_selection(
                    evaluated_population, tournament_size)
                parent2 = tournament_selection(
                    evaluated_population, tournament_size)

                child1, child2 = None, None

                # 4. Cruzamento (com crossover_rate)
                if random.random() < crossover_rate:
                    child1, child2 = two_point_crossover(parent1, parent2)
                else:
                    child1, child2 = list(parent1), list(parent2)

                # 5. Mutação (com mutation_rate)
                mutated_child1 = mutate_individual(
                    child1, mutation_rate, max_possible_trip_for_mutation)
                mutated_child2 = mutate_individual(
                    child2, mutation_rate, max_possible_trip_for_mutation)

                # Otimização local dos filhos
                optimized_child1 = check_and_merge_trips(
                    mutated_child1, item_population_volumes, truck_capacity)
                optimized_child2 = check_and_merge_trips(
                    mutated_child2, item_population_volumes, truck_capacity)

                # Adicionar filhos à próxima geração (garantir limite de population_size)
                if len(next_generation_population) < population_size:
                    next_generation_population.append(optimized_child1)
                if len(next_generation_population) < population_size:
                    next_generation_population.append(optimized_child2)

            current_population = next_generation_population
            current_generation += 1
            ga_step_counter += 1

            # Imprimir progresso no console
            if current_generation % 10 == 0 or current_generation == 1:
                print(
                    f"Geração: {current_generation}/{num_generations}, Melhor Fitness: {best_overall_fitness}")
                print(
                    f"Gerações sem melhoria: {generations_without_improvement}")

        # --- Lógica de Desenho do Pygame ---
        screen.fill(WHITE)  # Limpar tela

        # Exibir informações do GA
        gen_text = font.render(
            f"Geração: {current_generation}/{num_generations}", True, BLACK)
        fitness_text = font.render(
            f"Melhor Fitness (Viagens): {best_overall_fitness}", True, BLACK)
        screen.blit(gen_text, (10, 10))
        screen.blit(fitness_text, (10, 40))

        # Visualizar a Melhor Solução Global, se disponível
        if best_overall_solution and best_overall_fitness != float('inf'):
            display_solution_text = big_font.render(
                "Melhor Solução Atual:", True, BLUE)
            screen.blit(display_solution_text, (10, 80))

            # Parâmetros de desenho para caminhões e itens
            truck_width = 150
            truck_height = 80
            truck_padding = 20
            start_x = 20
            start_y = 120

            current_x = start_x
            current_y = start_y

            sorted_trip_numbers = sorted(best_overall_trip_volumes.keys())

            for trip_num in sorted_trip_numbers:
                trip_vol = best_overall_trip_volumes[trip_num]

                # Desenhar Caminhão
                truck_rect = pygame.Rect(
                    current_x, current_y, truck_width, truck_height)
                pygame.draw.rect(screen, TRUCK_COLOR, truck_rect, 0)
                pygame.draw.rect(screen, BLACK, truck_rect, 2)

                # Exibir número da viagem e volume atual/capacidade
                trip_info_text = font.render(
                    f"Viagem {trip_num}: {trip_vol}/{truck_capacity}", True, BLACK)
                screen.blit(trip_info_text, (current_x, current_y - 25))

                # Visualizar itens dentro do caminhão (simplificado)
                fill_percentage = min(1.0, trip_vol / truck_capacity)
                fill_width = int(truck_width * fill_percentage)
                fill_rect = pygame.Rect(
                    current_x, current_y + truck_height - 10, fill_width, 10)

                fill_color = GREEN if fill_percentage <= 0.9 else (
                    255, 165, 0)  # Laranja se quase cheio
                if fill_percentage > 1.0:  # Não deve acontecer se a solução for válida
                    fill_color = RED
                pygame.draw.rect(screen, fill_color, fill_rect)

                # Mover para a próxima posição do caminhão
                current_x += truck_width + truck_padding
                if current_x + truck_width > SCREEN_WIDTH:
                    current_x = start_x
                    current_y += truck_height + truck_padding + 30

        if ga_step_counter >= num_generations:
            final_text = big_font.render("GA Finalizado!", True, RED)
            screen.blit(final_text, (SCREEN_WIDTH // 2 -
                        final_text.get_width() // 2, SCREEN_HEIGHT - 50))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


# --- Bloco de execução principal para rodar o GA com Pygame ---
if __name__ == "__main__":
    # Definir parâmetros do GA
    NUM_TOTAL_ITEMS = 100
    MAX_ITEM_VOL = 20
    MAX_TOTAL_VOLUME = 55000
    TRUCK_CAPACITY = 100
    POPULATION_SIZE = 200  # Aumentado para maior diversidade
    NUM_GENERATIONS = 1000000
    CROSSOVER_RATE = 0.85
    MUTATION_RATE = 0.08
    TOURNAMENT_SIZE = 4
    NUM_ELITE_INDIVIDUALS = 3  # Aumentado para preservar mais soluções boas

    print("Iniciando Algoritmo Genético com Visualização Pygame...")
    run_genetic_algorithm(
        NUM_TOTAL_ITEMS,
        MAX_ITEM_VOL,
        MAX_TOTAL_VOLUME,
        TRUCK_CAPACITY,
        POPULATION_SIZE,
        NUM_GENERATIONS,
        CROSSOVER_RATE,
        MUTATION_RATE,
        TOURNAMENT_SIZE,
        NUM_ELITE_INDIVIDUALS  # Passando o parâmetro de elitismo
    )
    print("\nAlgoritmo Genético finalizado.")
