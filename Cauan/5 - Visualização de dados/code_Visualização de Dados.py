# %%
# Base para o código:

## Leitura de arquivos e interpertação de dados:
import numpy as np 
import pandas as pd

## Importação e preparação dos arquivos
covid_19 = pd.read_csv(r"\Users\Cauan\Documents\UFV\NIAS-IA-treinamento-2024-2\Cauan\4 - Pandas\COVID-19 Dataset\covid_19_clean_complete.csv", index_col="Date", parse_dates=True)
worldmeter = pd.read_csv(r"\Users\Cauan\Documents\UFV\NIAS-IA-treinamento-2024-2\Cauan\4 - Pandas\COVID-19 Dataset\worldometer_data.csv")

print('Setup completo :)')

# %% [markdown]
# # 5.1 - Line Chart

# %% [markdown]
# ### 1. Criar gráfico linechart que indique a progressão no tempo das mortes nas regiões: Américas, Eastern Mediterran e Europe (OBS: Utilize apenas o arquivo “covid_19_clean_complete.csv”)

# %%
# a. Importar bibliotecas matplotlib e seaborn, para retirar a necessidade de criar arquivos para os gráficos, digitar comando “%matplotlib inline”;

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings # To suppress some warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

# %%
# b. Realizar agrupamento utilizando as colunas “Date” e “WHO Region”, utilizando “sum” como função de agregação;

sel_regions = covid_19.loc[(covid_19['WHO Region'] == 'Europe')|(covid_19['WHO Region'] == 'Eastern Mediterranean')|(covid_19['WHO Region'] == 'Americas')]
                                                                                                                
deaths_data = sel_regions.groupby(['Date', 'WHO Region']).agg({'Deaths':'sum'})

deaths_data.head()

# %%
# c. Plotar gráficos que contenham as progressões das regiões exigidas;

plt.figure(figsize=(14,6))

# Adicionar nomes para eixos, título do gráfico, grid e legendas para as linhas:

plt.title('Time progression for deaths by time in the regions: Americas, Eastern Mediterranean and Europe')

sns.lineplot(data=deaths_data, x='Date', y='Deaths', hue='WHO Region')

plt.show()

# %% [markdown]
# 
# Analisar gráfico de acordo com as perguntas feitas.
# 
#     1 - Europa e Américas
#     2 - Em meados de março
#     3 - De forma linear para as Américas, logaritmica para a Europa e de forma exponencial, apesar de muito lenta, para o Mediterrâneo Oriental
# 

# %% [markdown]
# # 5.2 - Bar Charts

# %% [markdown]
# ### 1. Agora deve-se produzir um gráfico bar chart, indicando o número de mortes em cada continente (OBS: utilize o dataframe criado no exercício 4.1 - 4)

# %%
#Recriando o dataframe do item 4.1 - 4

worldmeter_country = worldmeter.groupby('Country/Region').agg({'Continent':'sum','Population':'sum'})

covid_country = covid_19.groupby(['Country/Region','WHO Region'],as_index=False).agg({'Deaths':'sum'}).set_index('Country/Region')

world_covid = worldmeter_country.join(covid_country.loc[:,['Deaths']])

continentmeter = world_covid.reset_index().groupby('Continent').agg({'Country/Region': ', '.join,'Population':'sum','Deaths':'sum'})

continentmeter

# %%
# a. Editar o tamanho do gráfico, como sugestão, coloque o tamanho (10,6);

plt.figure(figsize=(10,6))

# %%
# b. Plotar gráfico bar chart das mortes nos continentes;

plt.title('Comparação Entre as Quantidades de Mortes nos Continentes')

sns.barplot(x=continentmeter.reset_index().Continent,y=continentmeter.reset_index().Deaths)

plt.xlabel('Continents')

plt.show()

# %% [markdown]
# 
# Responder perguntas com base no gráfico.
# 
#     1 - Europa e América do Sul
#     2 - Australia/Oceania
#     3 - Europa e América do Sul foi o continente que misturou 3 características que influenciaram muito na reprodução do vírus: ignorância geral quando o vírus primeiro chegou, alta densidade de pessoas e alto fluxo de pessoas. Em contrapartida, Australi/Oceania, se beneficiaram exatamente por características opostas: distanciamento geográfio, maior preoparo da população e baixo fluxo de pessoas.
# 
# 

# %% [markdown]
# # 5.3 - Scatter Plots

# %% [markdown]
# ### 1. Descobrir se há relação entre o número de mortes por milhão e a quantidade de habitantes, nos 3 continentes com maior média de mortes por milhão de habitantes, por meio de scatter plot (OBS: Utilizar somente arquivo “worldometer_data.csv”)

# %%
# a. Agrupar o dataframe pela coluna “Continent”, retirar apenas os dados de "Deaths/1 M pop", utilizar “mean” como função de agregação
# e ordenar para descobrir os 3 continentes com maior valor;

top_3 = worldmeter.groupby('Continent').agg({'Deaths/1M pop':'mean'}).sort_values(by='Deaths/1M pop', ascending=False).iloc[0:3]

top_3

# %%
# b. Obter apenas os dados do dataframe, correspondentes a esses continentes;

top_3_data = worldmeter.reset_index().loc[worldmeter.reset_index().Continent.isin(top_3.index)]

top_3_data

# %%
# c. Criar Scatterplot sendo o eixo x mostrando as mortes por milhão, o eixo y, a população total e representar os 3 continentes com cores diferentes;

sns.lmplot(x='Deaths/1M pop', y="Population", hue="Continent", data=top_3_data)

plt.title('Relação Entre o Número de Mortes por Milhão e a Quantidade de Habitantes')
plt.xlabel('Deaths/1 M pop')

plt.show()

# %% [markdown]
# 
# Responder as perguntas com base no gráfico.¶
# 
#     1 - No canto esquerdo inferior, devido a baixa população de alguns países em conjunto com a escala escolhida pelo seaborn
#     2 - Sim, uma fraca correlação positiva
#     3 - Sim, para os 3 continentes é possível perceber uma correlação positiva entre população e morte por milhão
# 
# 

# %% [markdown]
# # 5.4 - Heatmap de Correlação

# %% [markdown]
# ### 1. Será utilizado o método DataFrame.corr() do pandas para gerar os valores de correlação, então o aluno deverá produzir um heatmap para a visualização dos valores gerados

# %%
# a. Criar uma variável para armazenar os valores de correlação;

world_corr = worldmeter.corr(numeric_only=True)
world_corr

# %%
# b. Criar um heatmap, com título, utilizando a variável criada no item anterior;

plt.figure(figsize=(10,6))
plt.title('Correlação Entre os Dados Númericos Presentes no Dataframe')

sns.heatmap(data=world_corr, annot=True)

plt.show()

# %% [markdown]
# Escolher duas features para analisar seus valores de correlação (Sugestão: “Population”, “Teste/1M pop”).
# 
# Note que, ao analisar a correlação entre a População e a quantidade de testes por milhão de pessoas, notamos uma fraca correlação negativa. Isso significa que, quanto maior a população, menos testes são feitos.


