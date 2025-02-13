# %%
# Importar as bibliotecas que irão ser utilizadas:

# Para leitura de dados:
import numpy as np
import pandas as pd

# Para criação de gráficos e visualização de informações:
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings # To suppress some warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

# Para montagem do modelo
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
np.random.seed(0)

print('Setup completo :)')

# %% [markdown]
# # 10.1 - Exploração do banco de dados

# %% [markdown]
# ### 1. O primeiro passo é importar e visualizar o banco de dados, rastrear os valores faltantes e identificar os momentos estatísticos existentes:

# %%
# a. Importar o banco de dados utilizando o pandas;
unf_train_data = pd.read_csv('/kaggle/input/szeged-weather/weatherHistory.csv')
unf_train_data.head()

# %%
# b. Identificar o número de valores faltantes em cada colunas;
unf_train_data.isnull().sum()

## i. Qual sua hipótese para a existência desses valores faltantes?
## > Não houve precipitação no dia em questão

# %%
# c. Gerar momentos estatísticos do banco de dados;

unf_train_data.describe()

## i. Qual coluna parece fora do comum? Por que?
## > a coluna 'Loud Cover' aparenta não possuir valor algum, totalmente fora do padrão do resto do dataset

# %%
# d. Utilizando o método “.hist” do pandas realize plots de distribuição dos valores das colunas:

fig, ax = plt.subplots(4,2, figsize=(15,12))
fig.tight_layout()

unf_num_cols = unf_train_data.select_dtypes(['int64','float64']).columns

n = 0

for i in [0,1]:
    for j in [0,1,2,3]:
        sns.histplot(unf_train_data[unf_num_cols[n]], ax = ax[j,i])
        n += 1

## i. É possível identificar algo de estranho com a coluna “Pressure(millibars)”. Identifique e explique o que há de errado;
## > Através do gráfico, e também do describe() do dataset, é possível observar alguns dados 'outliers' com valores extremamente baixos

## ii. Utilizando “.loc” do bandas, identifique a quantidade de valores errados.

print('Existem {} valores na coluna Pressure (millibars) que estão errados'.format(unf_train_data.loc[unf_train_data['Pressure (millibars)'].lt(900)].count()[0]))

# %% [markdown]
# ### 2. Deve-se agora definir quais são as features categóricas e quais são as numéricas, para isso existe o método “.select_dtypes”, aqui será necessário identificar apenas os nomes das features.

# %%
# a. Identifique as features numéricas, aquelas que são do tipo “int64” ou “float64”, utilizando “.select_dtypes”;

unf_num_cols = unf_train_data.select_dtypes(['int64','float64']).columns

# %%
# b. Identifique as features categóricas, aquelas que são do tipo “object”;

unf_obj_cols = unf_train_data.select_dtypes('object').columns

## i. Uma das features é do tipo object mas não é categórica. Qual é ela?
## > A feature Formatted Date

unf_train_data.dtypes

# %%
# c. Inspecione os valores únicos de cada feature categórica, para garantir que não há nenhum valor que ocorra apenas uma vez,
# o que iria interferir na separação do banco de dados em treino e validação (Por que?). Utilize o método ".value_counts" para isso;
# > Influenciaria no fit do modelo, pois o valor único poderia cair no set de validação, impedindo que o modelo seja corretamente
# ajustado (fit).

for i in unf_obj_cols.drop('Formatted Date'):
    print('Valores únicos na feature', i,': \n', unf_train_data[i].value_counts().loc[unf_train_data[i].value_counts().lt(2)], '\n')

## i. Há uma feature em que existem valores que não ocorrem mais de uma vez. esse valores serão retirados do banco de dados futuramente;

# %% [markdown]
# # 10.2 - Limpeza do banco de dados

# %% [markdown]
# ### 1. Uma das features do tipo “object” é “Formatted Date” que, na verdade, indica a data do registro climático. Essa coluna deverá então ser transformada para o formato de data para que seja possível extrair informações úteis para o treinamento do modelo.

# %%
# a. Existe um valor nessa feature que indica o fuso-horário (“+0200”). Para que a coluna seja corretamente transformada este valor desse ser retirado;

## i. Deve ser realizado um processo semelhante ao item 9.2-1, em que será utilizado o método “str.split” para retirar o texto “+0200” de todos
## os valores da feature;

unf_train_data['Formatted Date'] = unf_train_data['Formatted Date'].str.split('+', n=1, expand=True)[0]

## ii. Use o método "to_datetime" para transformar a coluna no formato desejado.

unf_train_data['Formatted Date'] = pd.to_datetime(unf_train_data['Formatted Date'])

unf_train_data.dtypes

# %%
# b. Utilizando os métodos existentes em “.dt” crie novas colunas retirando as informações de hora, dia, mês e ano de cada valor de data na feature;

## i. Utilize os métodos “dt.hour ”, “dt.day ”, “dt.month ”, “dt.year ”;

unf_train_data['Hour'] = unf_train_data['Formatted Date'].dt.hour
unf_train_data['Day'] = unf_train_data['Formatted Date'].dt.day
unf_train_data['Month'] = unf_train_data['Formatted Date'].dt.month
unf_train_data['Year'] = unf_train_data['Formatted Date'].dt.year

## ii. Retire a coluna original de data do banco de dados.

unf_train_data.drop('Formatted Date', axis=1, inplace = True)

# %% [markdown]
# ### 2. Agora será criada a pipeline para completar a limpeza dos dados, com ela será realizada a substituição dos valores faltantes nas colunas “Pressure (millibars)” e “Precip Type” e o encoding, utilizando ordinal encoder, das variáveis categóricas. Para isso será utilizada a biblioteca Column Transformer do sklearn, mas antes deverão ser retirados os dados que não servirão para o treinamento do modelo.

# %%
# a. Retirar a coluna “Loud Cover”, já que contém apenas valores nulos, e linhas cujo valor da coluna “Summary” aparecem apenas uma vez;

train_topip = unf_train_data.copy()

## i. Para localizar as linhas que serão retiradas utilize o método “.loc”.

train_topip.drop('Loud Cover', axis=1, inplace=True)
train_topip.drop(train_topip.loc[(train_topip['Summary'] == 'Windy and Dry') | (train_topip['Summary'] == 'Dangerously Windy and Partly Cloudy') | (train_topip['Summary'] == 'Breezy and Dry')].index, inplace=True)

train_topip.dtypes

# %%
# b. Definir imputers para variáveis numéricas e para categóricas, e o encoder das variáveis categóricas;

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

topip_num_cols = train_topip.select_dtypes(['int64','float64','int32']).columns
topip_cat_cols = train_topip.select_dtypes('object').columns

## i. Configurar imputer das variáveis numéricas com simple imputer, utilizando a média como substituta do valor faltante e indicando que o
## valor 0 significa valor faltante (valores iguais a 0 em “Pressure (millibars)” representam valores faltantes);

num_imputer = SimpleImputer(strategy='mean', missing_values = 0)

## ii. Criar uma pipeline para tratamento das variáveis categóricas, primeiro passo deve ser a substituição dos valores faltantes utilizando simple
## imputer, substituindo o valor faltante por um texto, o segundo passo é realizar o encoding utilizando ordinal encoder.

OEnc = OrdinalEncoder()

cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant')),('encoder', OEnc)])

# %%
# c. Criar um objeto do tipo Column Transformer para aplicar os passos criados anteriormente.

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([('num', num_imputer, list(topip_num_cols)),('cat', cat_transformer, list(topip_cat_cols.drop('Daily Summary')))], remainder='passthrough')

# %% [markdown]
# ### 3. O último passo da limpeza dos dados será a separação em treino e validação, aplicação da pipeline de limpeza e o encoding da coluna que será o target (“Daily summary”).

# %%
# a. Definir quais são as features, o que é o target e separar o banco de dados utilizando train-test split,
# deve-se garantir que todos os valores do target estão distribuídos de forma equilibrada na reparação do banco de dados;

## i. Indicar o target como sendo “Daily Summary” e as features como as colunas restantes;

X = train_topip.copy()
y = X.pop('Daily Summary')

## ii. Utilizar train-test split, configurado para que o tamanho do banco de dados de teste seja 30% do banco de dados original e utilizando o
## target como parâmetro de “stratify”;

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1, test_size=0.3, stratify = y)

train_X

# %%
# b. Aplicar pipeline de limpeza do banco de dados;

## i. Utilize apenas as features dentro da pipeline, ajuste utilizando as features de treino e apenas transforme as de teste.

cltrain_X = preprocessor.fit_transform(train_X)
clval_X = preprocessor.transform(val_X)

data_final = pd.DataFrame(cltrain_X, columns = list(topip_num_cols) + list(topip_cat_cols.drop('Daily Summary')))

data_final

# %%
# c. Utilize label encoder para realizar o encoding do target.

from sklearn.preprocessing import LabelEncoder
LEnc = LabelEncoder()

## i. Da mesma forma, utilize o target de treino para ajustar e apenas transforme o de teste.

enctrain_y = LEnc.fit_transform(train_y)
encval_y = LEnc.transform(val_y)

# %% [markdown]
# # 10.3 - Treinamento e Avaliação da rede neural

# %% [markdown]
# ### 1. Primeiramente deve-se criar a arquitetura da rede neural, ela terá duas camadas escondidas, sendo cada uma com uma saída de dimensão 256, entre elas serão utilizados Batch normalization e Dropout. Como funções de ativação serão usados “relu” para as camadas escondidas e o “softmax” na camada de saída, esta função serve para que o modelo realize predições na forma de classificação não binária, ou seja, que seja capaz de prever mais de duas classes.

# %%
# a. Configurar o tamanho da entrada, que é o número de features para treinamento, e o tamanho da saída, que será a quantidade de classes do
# target;

from tensorflow import keras
from tensorflow.keras import layers

input_shape = X.shape[1]
print('Tamanho da entrada: ', input_shape)

output_shape = y.nunique()
print('Tamanho da saída: ', output_shape)

# %%
# b. Construir arquitetura do modelo utilizando a descrição já informada;

model = keras.Sequential([
    
    ## i. Camada de entrada utilizando batch normalization configurando seu tamanho a partir das features do banco de dados;

    layers.BatchNormalization(),

    ## ii. Primeira camada escondida utilizando “relu” como função de ativação, saída de dimensão 256, batch normalization e dropout de 0.3;

    layers.Dense(256, activation='relu', input_shape=input_shape),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.3),

    ## iii. Segunda camada escondida igual à primeira;

    layers.Dense(256, activation='relu', input_shape=input_shape),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.3),

    ## iv. Camada de saída com tamanho igual ao número de classes do target e função de ativação “softmax”.

    layers.Dense(output_shape, activation='softmax'),

])

# %%
# c. Realizar tratamento do target utilizando o método “utils.to_categorical” do keras.

## i. Transformar target de treino e de teste utilizando o método do keras;

ktrain_y = keras.utils.to_categorical(enctrain_y)
kval_y = keras.utils.to_categorical(encval_y)

## ii. Veja as dimensões da matriz de target e responda, o target é composto de quantas classes (Possíveis previsões)?

ktrain_y

# %% [markdown]
# ### 2. Configurar o otimizador, a função de perda que será utilizada e a métrica para avaliação do modelo.

# %%
# a. Como otimizador será utilizado “adam”, como função de perda “categorical_crossentropy” e como métrica para avaliação
# “categorical_accuracy”.

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy'],
)

# %% [markdown]
# ### 3. Por fim, será realizado o treinamento do modelo e plotagem dos resultados das avaliações, será utilizado o early stopping para evitar o overfitting. Os hiperparâmetros que serão utilizados, como o número de épocas e tamanho de cada batch durante o treinamento do modelo podem ser alterados pelo aluno para teste e possível aprimoramento.

# %%
# a. Configurar early stopping;

## i. Utilizar a biblioteca do keras para configurar early stopping, com uma tolerância de 10 épocas e uma variação mínima no resultado da função
## de perda de 0.001.

early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
)

cltrain_X

# %%
# b. Realizar o treino e armazenar os resultados da avaliação do modelo;

history = model.fit(

    ## i. Utilizar as features e target de treino para treinar o modelo;
    cltrain_X, ktrain_y,

    ## ii. Utilizar os dados de teste para validar o modelo;
    validation_data=(clval_X, kval_y),

    ## iii. Utilizar um tamanho de batch de 3000;
    batch_size=3000,

    ### iv. Utilizar 100 épocas para treino;
    epochs=100,

    ### v. Indicar utilização de early stopping;
    callbacks=[early_stopping],

    verbose=0, # esconde o texto das épocas
)

# %%
# c. Para a plotagem deve-se transformar os resultados do treino em um dataframe o pandas e plotar os valores de perda e acurácia em função das épocas.

## i. Utilizar o método “.DataFrame” para transformar resultados em data frame do pandas;

history_df = pd.DataFrame(history.history)

## ii. Utilizar gráfico de linha para plotar os resultados da perda e da acurácia em função das épocas de treino.

history_df.head()

history_df.loc[5:, ['loss', 'val_loss']].plot()
history_df.loc[5:, ['categorical_accuracy', 'val_categorical_accuracy']].plot()

print(("Best Validation Loss: {:0.4f}" +\
      "\nBest Validation Accuracy: {:0.4f}")\
      .format(history_df['val_loss'].min(), 
              history_df['val_categorical_accuracy'].max()))


