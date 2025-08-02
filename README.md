# Otimizador de Rotas de Entrega com Algoritmo Genético

## Descrição
Este projeto implementa um algoritmo genético para otimizar rotas de entrega de produtos, minimizando o número de viagens necessárias considerando restrições de capacidade do caminhão.

## Características Principais
- Interface gráfica interativa usando Pygame
- Otimização através de algoritmo genético
- Visualização em tempo real da evolução das soluções
- Capacidade de salvar e carregar populações de teste

## Pré-requisitos
- Python 3.13 ou superior
- Bibliotecas necessárias:
  - pygame
  - numpy
  - matplotlib

## Instalação
1. Clone o repositório
2. Crie um ambiente virtual Python:
```bash
python -m venv genetic_algorithm
source genetic_algorithm/bin/activate  # Linux/Mac
```
3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Estrutura do Projeto
```
.
├── src/
│   ├── main.py                    # Arquivo principal com interface gráfica
│   ├── genetic_methods.py         # Implementação do algoritmo genético
│   └── generate_fixed_population.py# Gerador de população de teste
├── fixed_population.json          # População de teste salva
├── requirements.txt               # Dependências do projeto
└── README.md                      # Este arquivo
```

## Como Usar
1. Ative o ambiente virtual:
```bash
source genetic_algorithm/bin/activate
```

2. Para gerar uma nova população de teste:
```bash
python src/generate_fixed_population.py
```

3. Para executar o otimizador:
```bash
python src/main.py
```

## Parâmetros do Algoritmo Genético
- População: 300 indivíduos
- Taxa de Crossover: 0.85
- Taxa de Mutação: 0.8
- Tamanho do Torneio: 2
- Número de Indivíduos Elite: 2
- Capacidade do Caminhão: 100
- Número Máximo de Gerações: 1.000.000

## Visualização
A interface gráfica mostra:
- Número da geração atual
- Melhor fitness encontrado
- Representação visual dos caminhões e suas cargas
- Lista detalhada de itens por viagem
- Indicadores de utilização do caminhão (verde: ok, laranja: quase cheio, vermelho: sobrecarga)

## Como Funciona
1. **Geração da População**: Cria soluções iniciais aleatórias respeitando restrições
2. **Avaliação**: Calcula o fitness de cada solução considerando número de viagens e utilização
3. **Seleção**: Usa torneio para escolher indivíduos para reprodução
4. **Crossover**: Combina soluções para gerar novos indivíduos
5. **Mutação**: Aplica pequenas alterações para manter diversidade
6. **Elitismo**: Preserva as melhores soluções entre gerações


## Autores
- FIAP Pós-Graduação
- Disciplina: Inteligência Artificial e Aprendizado de Máquina

## Licença
Este projeto está sob a licença MIT.