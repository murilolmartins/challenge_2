# Generate Fixed Population (generate_fixed_population.py)

Este módulo é responsável por gerar e gerenciar uma população fixa para testes do algoritmo genético.

## Funções

### save_population_to_file
```python
save_population_to_file(population: list, filename="fixed_population.json")
```
Salva uma população gerada em um arquivo JSON.

**Parâmetros:**
- `population` (list): Lista de inteiros representando os volumes de cada item
- `filename` (str): Nome do arquivo onde salvar a população (padrão: "fixed_population.json")

**Comportamento:**
- Salva a lista de volumes em formato JSON com indentação de 4 espaços
- Exibe mensagem de sucesso ou erro no console
- Trata possíveis erros de I/O durante o salvamento

### load_population_from_file
```python
load_population_from_file(filename="fixed_population.json") -> list
```
Carrega uma população previamente salva de um arquivo JSON.

**Parâmetros:**
- `filename` (str): Nome do arquivo de onde carregar a população (padrão: "fixed_population.json")

**Retorno:**
- Lista de inteiros representando os volumes dos itens
- Lista vazia em caso de erro

**Tratamento de Erros:**
- Verifica se o arquivo existe
- Trata erros de decodificação JSON
- Trata erros de I/O durante a leitura

### Execução Principal
Quando executado como script principal (`__main__`), o módulo:
1. Gera uma nova população com 200 itens
2. Volume máximo por item: 50
3. Salva a população no arquivo padrão

## Exemplo de Uso
```python
# Gerar e salvar nova população
population = generate_population(population_size=200, max_volume_per_individual=50)
save_population_to_file(population)

# Carregar população existente
loaded_population = load_population_from_file()
```

## Arquivos
- **Entrada**: Nenhum (ao gerar nova população)
- **Saída**: fixed_population.json (arquivo JSON com a população gerada)

## Dependências
- random: Geração de números aleatórios
- json: Manipulação de arquivos JSON
- os: Operações do sistema de arquivos
- genetic_methods: Módulo com funções do algoritmo genético