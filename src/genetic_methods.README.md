# Métodos Genéticos (genetic_methods.py)

Este módulo implementa as funções core do algoritmo genético para otimização de rotas.

## Funções Principais

### generate_population
```python
generate_population(population_size, max_volume_per_individual=10) -> list
```
Gera uma população inicial aleatória de volumes.

**Parâmetros:**
- `population_size`: Número de indivíduos
- `max_volume_per_individual`: Volume máximo por item

### generate_max_number_of_travels
```python
generate_max_number_of_travels(population, truck_volume)
```
Calcula os limites teóricos de número de viagens.

**Retorno:**
- Lista `[min_viagens_otimista, max_viagens_pessimista]`

### calculate_fitness
```python
calculate_fitness(individual_plan, all_item_volumes, truck_capacity)
```
Avalia a qualidade de uma solução.

**Considera:**
- Número de viagens necessárias
- Utilização do caminhão
- Penalidades por violações
- Distribuição da carga

### tournament_selection
```python
tournament_selection(population_with_fitness, k=3) -> list
```
Seleciona indivíduos para reprodução via torneio.

### two_point_crossover
```python
two_point_crossover(parent1, parent2)
```
Realiza cruzamento de dois pontos entre pais.

### mutate_individual
```python
mutate_individual(individual, mutation_rate, max_possible_trip_number)
```
Aplica mutações com três estratégias:
- Mutação suave (40%): ±1 viagem
- Mutação média (30%): ±3 viagens
- Mutação forte (30%): Viagem aleatória

### diversify_population
```python
diversify_population(population, item_volumes, truck_capacity, mutation_rate, max_possible_trip_number)
```
Diversifica população estagnada:
- Mantém 10% melhores
- Aplica estratégias variadas de diversificação
- Completa com indivíduos aleatórios se necessário

### check_and_merge_trips
```python
check_and_merge_trips(individual, item_volumes, truck_capacity)
```
Otimiza solução combinando viagens compatíveis.

### fix_trip_sequence
```python
fix_trip_sequence(individual)
```
Normaliza numeração das viagens (remove "buracos").

### Funções Auxiliares

#### random_shuffle_segments
```python
random_shuffle_segments(individual)
```
Embaralha segmentos do cromossomo.

#### invert_segments
```python
invert_segments(individual)
```
Inverte segmentos aleatórios.

## Estruturas de Dados

### Indivíduo (Cromossomo)
- Lista de inteiros
- Cada posição (gene) representa item
- Valor indica número da viagem

### Fitness (Avaliação)
Dicionário contendo:
- 'fitness': Valor numérico (menor = melhor)
- 'is_valid': Booleano indicando validade
- 'trip_volumes': Volumes por viagem

## Estratégias de Otimização

1. **Penalização Gradual**
   - Excesso de capacidade
   - Baixa utilização
   - Distribuição desigual

2. **Diversificação**
   - Mutação agressiva
   - Embaralhamento de segmentos
   - Inversão de segmentos

3. **Elitismo**
   - Preservação dos melhores
   - Recombinação controlada
   - Mutação adaptativa

## Dependências
- random: Operações aleatórias
- math: Cálculos matemáticos