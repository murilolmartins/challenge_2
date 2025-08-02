# Interface Gráfica do Algoritmo Genético (main.py)

Este módulo implementa a interface gráfica e controla a execução do algoritmo genético.

## Função Principal

### run_genetic_algorithm
```python
run_genetic_algorithm(
    num_items,
    max_item_volume,
    truck_capacity,
    population_size,
    num_generations,
    crossover_rate,
    mutation_rate,
    tournament_size=3,
    num_elite_individuals=1
)
```

**Parâmetros:**
- `num_items`: Total de itens para empacotar
- `max_item_volume`: Volume máximo por item
- `truck_capacity`: Capacidade do caminhão por viagem
- `population_size`: Tamanho da população
- `num_generations`: Número máximo de gerações
- `crossover_rate`: Taxa de crossover (0.0 a 1.0)
- `mutation_rate`: Taxa de mutação (0.0 a 1.0)
- `tournament_size`: Tamanho do torneio na seleção
- `num_elite_individuals`: Número de indivíduos elite

## Interface Gráfica

### Elementos Visuais
1. **Informações Gerais**
   - Geração atual/total
   - Melhor fitness encontrado

2. **Visualização de Caminhões**
   - Representação visual de cada viagem
   - Barra de preenchimento colorida:
     - Verde: Ocupação normal
     - Laranja: Quase cheio
     - Vermelho: Sobrecarga

3. **Lista de Itens**
   - Agrupados por viagem
   - Ordenados por volume
   - Formato: índice(volume)

### Dimensões e Layout
- Tela: 1920x1080
- Caminhões: 100x60 pixels
- Espaçamento: 25 pixels
- Fontes:
  - Normal: 16pt
  - Grande: 24pt
  - Pequena: 20pt (listas)

## Fluxo do Algoritmo

1. **Inicialização**
   - Carrega população do arquivo
   - Gera população inicial
   - Configura interface gráfica

2. **Loop Principal**
   - Avaliação da população
   - Detecção de estagnação
   - Evolução da população
   - Atualização da visualização

3. **Evolução**
   - Seleção por torneio
   - Crossover de dois pontos
   - Mutação adaptativa
   - Aplicação de elitismo

4. **Visualização**
   - Atualização em tempo real
   - Renderização de caminhões
   - Atualização de estatísticas

## Configurações Padrão
```python
NUM_TOTAL_ITEMS = 200
MAX_ITEM_VOL = 20
TRUCK_CAPACITY = 100
POPULATION_SIZE = 300
NUM_GENERATIONS = 1000000
CROSSOVER_RATE = 0.85
MUTATION_RATE = 0.8
TOURNAMENT_SIZE = 2
NUM_ELITE_INDIVIDUALS = 2
```

## Eventos e Controle
- Fechamento da janela: Encerra programa
- ESC: Encerra programa
- Atualização: 60 FPS (padrão Pygame)

## Dependências
- pygame: Interface gráfica
- genetic_methods: Funções do algoritmo genético
- generate_fixed_population: Carregamento de dados