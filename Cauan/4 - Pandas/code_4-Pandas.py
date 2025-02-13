# %%
# Base para o código:

## Leitura de arquivos e interpertação de dados:
import numpy as np 
import pandas as pd

## Importação e preparação dos arquivos
covid_19 = pd.read_csv(r"\Users\Cauan\Documents\UFV\NIAS-IA-treinamento-2024-2\Cauan\Pandas\COVID-19 Dataset\covid_19_clean_complete.csv")
worldmeter = pd.read_csv(r"\Users\Cauan\Documents\UFV\NIAS-IA-treinamento-2024-2\Cauan\Pandas\COVID-19 Dataset\worldometer_data.csv")

print('Setup completo :)')

# %% [markdown]
# # 4.1 - Explorando o "COVID-19 Dataset"

# %% [markdown]
# ### 1. O primeiro passo da exploração de dados é obter informações gerais, deve-se importar o arquivo “covid_19_clean_complete.csv” e obter as seguintes informações.

# %%
# a. Quantidade de linhas e colunas;

print('O dataset possui ',covid_19.shape[0],' linhas e ', covid_19.shape[1],' colunas')

# %%
# b. Quais são as colunas;

print('As colunas do dataset são: ', list(covid_19.columns))

# %%
# c. O tipo de dado de cada coluna;

covid_19.dtypes

# %%
# d. A coluna de datas deve ser transformada de “Object” para “datetime64”, utilizando o comando “pd.todatetime( )” ;

covid_19['Date']=pd.to_datetime(covid_19['Date'])

covid_19['Date'].dtype

# %%
# e. Informações estatísticas sobre o banco de dados;

covid_19.describe()

# %%
# f. Se necessário, transformar uma coluna pertinente para formato de data;

# Não há necessidade

# %%
# g. Quais colunas apresentam NaN.

covid_19.isnull().sum()

# %% [markdown]
# ### 2. Crie um dataframe que contenha apenas as 5 províncias da China que mais registraram casos.

# %%
# a. Descubra qual o nome das províncias chinesas contidas no dataset ;

c_provinces = covid_19.loc[covid_19['Country/Region'] == 'China']

c_provinces.groupby('Province/State')['Country/Region'].max()

# %%
# b. Retire do banco de dados apenas as informações dessas províncias;

c_provinces

# %%
# c. Tome apenas as informações das features: “Confirmed”, “Active”, “Deaths”, “Recovered”; 

c_provinces.loc[:,['Confirmed', 'Active', 'Deaths', 'Recovered']]

# %%
# d. Agrupe o novo dataframe a partir dessas províncias. Qual a função de agrupamento que deve ser usada?

provinces_cases = c_provinces.groupby('Province/State').agg({'Confirmed':'sum','Active':'sum','Deaths':'sum','Recovered':'sum'})

provinces_cases

# Foi usada o method '.groupby()' para montar o dataframe

# %%
# e. Produza um dataframe contendo as 5 regiões com o maior número de casos confirmados.

top_provinces = provinces_cases.sort_values(by='Confirmed', ascending=False).head()

top_provinces

# %% [markdown]
# ### 3. A coluna “Province/State” contém muitos valores faltando. Para descartá-la perdendo mínimo de informação, inclua na coluna “Country/Region” o nome das regiões junto ao nome dos países em que existem mais de uma província registrada no banco de dados, utilizando o método “.apply”. Como por exemplo, juntar o nome China com a região pertinente: “China_Hubei”, “China_Guangdong”.

# %%
# a. Elabore uma função que tenha uma linha do banco de dados como argumento e, se a coluna “Province/State” não for valor faltante,
# concatenar seu valor com coluna “Country/Region”.

def concat(row):
    
    ## i. Utilize o método “pandas.notna” para averiguar de “Province/State” é valor faltante.
    if pd.notnull(row['Province/State']):
        row['Country/Region'] = row['Country/Region'] + '_' + row['Province/State']

    return row

# %%
# b. Faça uma cópia do banco de dados para poder retirar informações sem perder o banco de dados original.

covid = covid_19.copy()

covid

# %%
# c. Aplique, por meio do método “apply”, a função criada no item a.

concat_covid = covid.apply(concat,axis='columns')

#Exemplo do resultado:

concat_covid.loc[concat_covid['Country/Region'] == 'China_Hubei']

# %%
# d. Exclua a coluna “Province/State” do novo dataframe.

concat_covid = concat_covid.loc[:,concat_covid.columns != 'Province/State']

#Exemplo do resultado final:

concat_covid.loc[concat_covid['Country/Region'] == 'China_Hubei']

# %% [markdown]
# ### 4. Importe o arquivo "world meter data.csv" e, a partir de seus dados de população e continentes, faça um ranking de maior número de mortes por milhão de habitantes entre os continentes.

# %%
# a. Importar o banco de dados "world meter data.csv".

worldmeter.head()

# %%
# b. Tome apenas as informações de população, país e continentes deste banco de dados.

worldmeter_country = worldmeter.groupby('Country/Region').agg({'Continent':'sum','Population':'sum'})

worldmeter_country

# %%
# c. Agrupe o “covid_19_clean_complete” por países. Deve-se usar o dataframe do item “c)” ou o original?

#Devemos utilizar o dataframe original, dado que o dataframe gerado no item c) não inclui informações de países

covid_country = covid_19.groupby('Country/Region').agg({'WHO Region': 'max','Deaths':'sum'})

covid_country

# %%
# d. Mescle os dois dataframes, a partir dos países, contendo apenas as informações de mortes, população e continente.

world_covid = worldmeter_country.join(covid_country.loc[:,['Deaths']])

world_covid.head()

# %%
# e. Agrupe o novo dataset agora pelo continente.

continentmeter = world_covid.reset_index().groupby('Continent').agg({'Country/Region': ', '.join,'Population':'sum','Deaths':'sum'})

continentmeter

# %%
# f. Crie uma nova coluna contendo o número de mortes por milhão dentro dos continentes (Utilize a fórmula: (mortes/população)*10^6).

dths_1m = []

for i in range(continentmeter.__len__()):
    dths_1m.append(((continentmeter.Deaths.iloc[i])/(continentmeter.Population.iloc[i]))*(10**6))

continentmeter['Deaths/1M_Pop']=dths_1m

continentmeter

# %%
# g. Faça um ranking do número de mortes por milhão de habitantes entre os continentes.

continentmeter.sort_values(by='Deaths/1M_Pop', ascending=False)


