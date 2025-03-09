# %% [markdown]
# # 6.1 - Importação e visualização do banco de dados

# %% [markdown]
# ### 1. Importar as bibliotecas que irão ser utilizadas:

# %%
# a. numpy as np;

import numpy as np

# %%
# b. pandas as pd;

import pandas as pd

# %%
# c. matplotlib.pyplot;

import matplotlib.pyplot as plt

# %%
# d. seaborn as sns;

import seaborn as sns
import warnings # To suppress some warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

# %%
# e. matplotlib inline, para gerar gráficos no próprio notebook;

%matplotlib inline

# %%
# f. Configurar o random seed para 0 (np.random.seed(0))

np.random.seed(0)

# %% [markdown]
# ### 2. Transformar arquivos em dataframe e gerar momentos estatísticos:

# %%
# a. Importar os arquivos “kaggle/input/titanic/test.csv” e “kaggle/input/titanic/train.csv”;

unf_train_data = pd.read_csv(r"C:\Users\Cauan\Documents\UFV\NIAS-IA-treinamento-2024-2\Cauan\6 - Introdução ao Machine Learning\Titanic Dataset\train.csv")
unf_test_data = pd.read_csv(r"C:\Users\Cauan\Documents\UFV\NIAS-IA-treinamento-2024-2\Cauan\6 - Introdução ao Machine Learning\Titanic Dataset\test.csv")

# %%
# b. Gerar momentos estatísticos sobre o dataframe de treino, utilizando “.describe( )”;

display('Momento estátistico do dataset de treino:',unf_train_data.describe())
display('Momento estátistico do dataset de teste:',unf_test_data.describe())

# %% [markdown]
# ### 3. Análise dos dados:

# %%
# a. Identificar as Features do banco de dados que são categóricas (não numéricas), e as que são numéricas. As numéricas,
# deve-se separá-las em discretas e contínuas.

## i. Pode-se usar o método “.info( )” para descobrir o tipo de dado de cada Feature.

unf_train_data.info()

# > Note que teremos como categóricas as features: Sex, Name, Ticket, Cabin e Embarked
# > Para as númericas, teremos por discretas as features: PassengerId, Survived, Pclass, SibSp e Parch. Por contínuas teremos: Age e Fare

# %%
# b. Descobrir quais as Features contém NaN, além da quantidade desse tipo de dado em cada Feature.

## i. Pode-se utilizar o comando “DataFrame.isnull.sum( )”.

unf_train_data.isnull().sum()

# %%
# c. Deve-se utilizar apenas Features numéricas e retirar as categóricas. Portanto, faça dois novos dataframes,
# apenas com as features numéricas do banco de dados de treino e de teste.

train_data = unf_train_data.groupby('PassengerId').agg({'Survived':'sum','Pclass':'sum','SibSp':'sum','Parch':'sum','Age':'sum','Fare':'sum'}).dropna()
test_data = unf_test_data.groupby('PassengerId').agg({'Pclass':'sum','SibSp':'sum','Parch':'sum','Age':'sum','Fare':'sum'}).fillna(value=0)

display(train_data)
display(test_data)

# %%
# As letras d. e e. já foram realizadas na formação dos novos datasets

# %% [markdown]
# #  6.2 - Exploração do banco de dados

# %% [markdown]
# ### 1. Para a Feature “Age” será necessário criar uma nova Feature que a separe em 8 grupos para que se possa utilizar “sns.barplot”. (DICA: Crie um novo dataframe apenas com as informações necessárias para a criação deste gráfico)

# %%
# a. criar uma lista com os nomes que serão dados aos grupos em que as idades serão repartidas (Exemplo: “(0-10)”, “(10-20)”, ...) ;

age_groups = ['(0-10)','(10-20)','(20-30)','(30-40)','(40-50)','(50-60)','(60-70)','(70-80)']

# %%
# b. utilize pandas.cut, para separar as idades em 8 grupos e utilize o argumento “labels” para utilizar a lista dos nomes dos grupos;
# c. criar a nova coluna com os grupos, perceba que esta nova coluna será categórica;

age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]

train_data['Age Groups']=pd.cut(train_data.Age, age_bins, right=False, include_lowest=True, labels=age_groups)

# %%
# d. criar um gráfico que indique a chance de sobrevivência por grupo de idade;

age_data = train_data.reset_index().groupby('Age_Groups').Survived.sum().reset_index()

plt.figure(figsize=(10,6))
plt.title('Sobreviventes por grupo de idade')

sns.barplot(x=age_data.Age_Groups,y=age_data.Survived)

plt.xlabel('Age Group')

plt.show()

# %% [markdown]
# Note que a regra de crianças e mulheres primeiros, em especial as crianças, que são representadas pelo grupo (0-10), surtiu amplo efeito na margem de sobrevivência destas.
# 
# Note também a taxa de sobrevivência caindo conforme o aumento da idade, implicando muito provavelmente no efeito da saúde e do vigor na sobrevivência, como também a equipe, provavelmente formada por adultos na faixa etária de 40-60.

# %% [markdown]
# ### 2. Crie gráficos para todas as outras features numéricas discretas, e faça uma análise dos resultados obtidos. Não será necessário produzir um gráfico para a feature “Fare”. Após a criação de cada gráfico será necessário realizar uma breve análise para avaliação do impacto de cada feature na predição.

# %%
# Parch

Parch_data = train_data.groupby('Parch').Survived.sum().reset_index()

plt.figure(figsize=(10,6))
plt.title('Sobreviventes por pais/filhos')

sns.barplot(x=Parch_data.Parch,y=Parch_data.Survived)

plt.xlabel('Parents/Children')

plt.show()

# Note que, similar ao gráfico acima, um número menor de pais ou filhos acompanhando o sobrevivente, maior a chance de sobrevivência. Nesse caso, a diferença é ainda mais
# drástica e muito disso certamente pode ser atrelado ao príncipio de mulheres e crianças primeiro nos botes, mas também novamente a difusão da atenção, maior fragilidade
# das crianças, maior fragilidade de pais idosos, etc.

# %%
#Pclass

Pclass_data = train_data.groupby('Pclass').Survived.sum().reset_index()

plt.figure(figsize=(10,6))
plt.title('Sobreviventes por número de nível da passagem')

sns.barplot(x=Pclass_data.Pclass,y=Pclass_data.Survived)

plt.xlabel('Ticket class')

plt.show()

# Note um padrão parecido entre as colunas, entretanto note a quantidade de passageiros por clase de ticket influencia muito nesse número, já que a quantidade de passageiros
# considerados terceira classe era consideravelmente maior do que os das classes "acima". Observando esse fato e a similaridade no número de sobreviventes, é possível
# observar as vantagens garantidas pelo nível da passagem do sobrevivente.

# %% [markdown]
# # 6.3 - Treinamento do modelo e submissão das predições

# %% [markdown]
# ### 1. Separação de banco de dados de treino e validação:

# %%
# a. Criar variável X, contendo o banco de dados apenas das Features;

train_features = ['SibSp','Pclass','Parch','Age', 'Fare']

X = train_data[train_features]

# %%
# b. Criar variável y, contendo o target da predição;

y = train_data.Survived

# %%
# c. Usar Train Test Split para separar treino e validação (DICA: utilizar validação com 20% do tamanho do treino);

from sklearn.model_selection import train_test_split

train_X,val_X, train_y,val_y = train_test_split(X,y,random_state=1,train_size=0.8)

# %% [markdown]
# ### 2. Treinamento e validação do modelo:

# %%
### a. Importar Random Forest Classifier da biblioteca sklearn.ensemble;

from sklearn.ensemble import RandomForestClassifier

# %%
# b. Importar accuracy score da biblioteca sklearn.metrics;

from sklearn.metrics import accuracy_score

# %%
# c. Treinar random forest com as features e o target de treino;

titanic_model = RandomForestClassifier(random_state=1)


titanic_model.fit(train_X,train_y)

# %%
# d. Realizar a predição;

titanic_model.predict(val_X)

# %%
# e. Validar a predição, comparando com o target da validação, utilizando accuracy score;

accuracy_score(val_y,titanic_model.predict(val_X))

# %% [markdown]
# Dada as limitações do modelo, a predição é satisfatória

# %% [markdown]
# ### 3. Gerar predição, salvar a versão e submeter à competição:

# %%
# Gerar predição, salvar a versão e submeter à competição:

titanic_model.fit(X,y)

index = test_data.index

test_X = test_data[train_features]

titanic_model.predict(test_X)

titanic_submission = pd.DataFrame({'PassengerId':index,'Survived':titanic_model.predict(test_X)}).set_index('PassengerId')
titanic_submission

titanic_submission.to_csv()

# %% [markdown]
# O código acima foi utilizado para gerar o arquivo de submissão em um notebook do Kaggle. O output resultante obteve precisão de 0.62200

# %% [markdown]
# # 7 - Data Cleaning

# %% [markdown]
# ### 1. Análise inicial:

# %%
# a. Quais as colunas do banco de dados de teste e de treino?

print('Colunas do dataset de treino: ', unf_train_data.columns,'\n Colunas do dataset de teste: ', unf_test_data.columns,'\n')

# %%
# b. Qual o tipo de dado de cada coluna nos dataframes de teste e de treino?

print('Tipos das colunas do dataset de treino:\n', unf_train_data.dtypes,'\n Tipos das colunas do dataset de test:\n', unf_test_data.dtypes,'\n')

# %%
# c. Qual a quantidade de valores nulos (NaN) em cada feature?

print('Valores nulos em cada coluna do dataset de treino:\n',unf_train_data.isnull().sum(), '\n Valores nulos em cada coluno do dataset de test:\n',unf_test_data.isnull().sum(),'\n')

# %%
# d. Realizar um cópia do banco de dados de teste e de treino para que se possa fazer a manipulação sem perder informações.

train_toclean = unf_train_data.copy()
test_toclean = unf_test_data.copy()

# %% [markdown]
# ### 2. Para lidar com valores nulos, podemos preencher estes valores de alguma forma ou descartar a informação. Neste item utilizaremos algumas estratégias para tal.

# %%
# Letras a., b., c. e d.:

train_rlvnt = train_toclean.drop(['Cabin','Ticket'], axis=1).fillna({'Age':train_toclean['Age'].median(),'Fare':train_toclean['Fare'].median(),'Embarked':train_toclean['Embarked'].mode()[0]})
test_rlvnt = test_toclean.drop(['Cabin','Ticket'], axis=1).fillna({'Age':train_toclean['Age'].median(),'Fare':train_toclean['Fare'].median(),'Embarked':train_toclean['Embarked'].mode()[0]})

# %% [markdown]
# ### 3. Para as Features contínuas será útil a criação de grupos para facilitar a análise. Dois métodos do pandas são úteis para esta tarefa, pd.cut e pd.qcut, também é útil visitar esta referência para uma melhor entendimento destes métodos.

# %%
# a. Criar Feature que separe a Feature “Age” em 5 intervalos de mesma extensão;

age_groups = ['(0-16)','(16-32)','(32-48)','(48-64)','(64-80)']
age_bins = [0, 16, 32, 48, 64, np.inf]

train_rlvnt['Age Groups']=test_rlvnt['Age Groups']=pd.cut(train_rlvnt.Age, age_bins, right=False, include_lowest=True, labels=age_groups)

# %%
# b. Criar Feature que separe “Fare” em 6 intervalos que contenham o mesmo número de dados (Não precisam ter a mesma extensão).

fare_groups = ['(0-4)','(4-8)','(8-16)','(16-32)','(32-64)','(64-513)']
fare_bins = [0,4,8,16,32,64,np.inf]

train_rlvnt['Fare Groups']=test_rlvnt['Fare Groups']=pd.cut(train_rlvnt.Fare, fare_bins, right=False, include_lowest=True, labels=fare_groups)


