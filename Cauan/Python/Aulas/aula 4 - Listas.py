###LISTAS###

#listas representam uma ordem sequenciada de valores
#elas são bem compreensivas e podem aceitar ints,
#floats, strings, até mesmo outras listas

planetas = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']

##INDEXAÇÃO##

#indexar significa acessar um elemento específico,
#nesse caso de uma lista. podemos fazer isso dando
# da lista e entre [] o valor da casa. Listas no
#Python tem a primeira casa no 0


print(planetas[0])

#Também podemos referenciar de trás pra frente, 
#começando do -1

print(planetas[-1])

##SEPARAÇÃO##

#podemos separar um pedaço da lista ao indicarmos
#o primeiro item a ser separado da lista até outro 
#ponto. O último ponto não é incluso na separação:

print(planetas[0:3])

#podemos também deixar o primeiro espaço vazio, e
#ele assumirá como o 0, ou o úlltimo espaço vazio
#e ele assumirá como o último item da lista:

print(planetas[:3])
print(planetas[3:])

 #também podemos usar números negativos para dividir:
 
print(planetas[1:-1]) #todos menos o primeiro e o último planeta
print(planetas[-3:]) #últimos 3 planetas

##ALTERAÇÕES NUMA LISTA##

#podemos alterar itens individuais de uma lista:

planetas[3]='Malacandra'
print(planetas)

#ou pedações dessa lista;

planetas[:3]=['Mur','Vee','Ur']
print(planetas)

#teste

planetas[-3:]='Plutão'
print(planetas)

planetas[-6:]=['Saturn', 'Uranus', 'Neptune']
print(planetas)

#pqp de alguma frma ele separa oq vc mandou ele
#colocar da forma mais doida possível

##FUNÇÕES DE LISTAS##

#len() nos dá o tamanho de uma lista

print(len(planetas))

#sorted() nos retorna a lista ordenada
#de uma dada forma:

print(sorted(planetas))

#para strings, é ordenado em ordem alfabética

#sum() soma os valores de uma lista:

primos = [2, 3, 5, 7]
print(sum(primos))

#min e max também funcionam em listas:

print(max(primos))

###INTERLÚDIO - OBJETOS###

#objetos são, essencialmente, tudo no python
#eles são coisas que carregam outras coisas com eles
#você pode acessar essas coisas com a sintaxe
# de pontos do python

#números, por exemplo, carregam consigo sua parte
#imaginária, seu tamanho em bits etc.:

x=1
print(x.bit_length())

#no caso de coisas como esse exemplo, uma função que
#acompanha um objeto, as chamaremos de 'methods'. Para
#para as não funções, como a parte imaginária, chamaremos
#de attributes.

##METHODS DAS LISTAS##

#'list'.append() adciona um elemento no final da lista

planetas.append('Plutão')
print(planetas)

#'list'.pop elimina o último item da lista:

planetas.pop()
print(planetas)

#'list'.index('item') nos da a indexação de um
#item em uma lista

print(planetas.index('Ur'))

#para esse último caso, para evitar erros, podemos
#usar o operator 'in' pra descobrir se alguma coisa 
#está em uma dada lista

#outros mpetodos não citados (que são muios) podem ser
#vistos com a ajuda da função help()

###TUPLAS###

#são essencialmente a mesma coisa que listas, com duas
#diferenças importantes:

#1.a criação delas usa parênteses aos invés de chaves:

t = (1, 2, 3)

#ou literalmente nada:

t = 1, 2, 3

#2.e a elas são IMUTÁVEIS

#tuplas são geralmente vistas em funções que retornam vários
#valores. Not que os valores de uma tupla podem ser individualmente
#associados de forma bastante prática:

a, b, c= t

print(a,b,c)

###FIM DA AULA###