###STRINGS E DICIONÁRIOS###

##SINTAXE##

help(dict)

#as principais notas a se tomar são que, usando '\'. podemos fazer alguns ajustes no texto. Por exemplo:

a = 'What\'s up?'
print(a, "o uso do \ permite o uso de um segundo ' dentro uma string já cercada por ' ")

b = 'Look, a mountain: /\\'
print(b, "usar \\ permite escrever \ no texto")

c = '1\n2 3'
print(c, 'usar \n passa o texto para a próxima linha')

#também podemos usar """ para cercar strings, permitindo que usemos o 'enter' ao inves de \n para criar uma nova linha:

d = """hello
world"""
print(d)

#por fim, print adciona naturalmente um \n no final de cada print, a não ser que seja alterado:

print("hello")
print("world")
print("hello", end='')
print("pluto", end='')

##STRINGS SÃO (QUASE) LISTAS##

#basicamente tudo que se aplica a uma lista, se aplica a um string, desde funções, até methods, até indexação, etc
#SALVO QUE STRINGS SÃO LISTAS IMUTÁVEIS

##METHODS DE STRINGS##

#além dos methods que uma lista teria, existem alguns methods específicos pra strings:

claim = "Pluto is a planet!"

print(claim.upper(), "string.upper() faz todas as letras da string ficarem em maiúsculo")

print(claim.lower(), "string.lower() faz todas as letras da string ficarem minúsculas")

print(claim.index('plan'), "string.index() mostra o index do primeiro elemento de uma substring")
#noe que, qualquer coisa em uma string é um elemento, até mesmo espaços e pontos, a não ser no caso da barra \ quando usada para
#permitir o uso de outro item

#\n conta como só um elemento

planet = 'Pluto'

print(claim.startswith(planet), "string.startswith() verifica se a string começa com a substring fornecida")

print(claim.endswith('planet'), "string.endswith() verifica se a string termina com a substring fornecida, nesse caso falso por causa da exclamação")

##SEPARANDO E ENTRANDO EM STRINGS##

#podemos dividir uma string em partes menores a partir de certos referenciais utilizando o method
#strig.split(). Por padrão, ela dividirá uma string em todos os seus espaços em branco

words = claim.split()
print(words)

datestr = '1956-01-31'
year, month, day = datestr.split('-')

print(year, month, day)

#string.join() faz o exato oposto

print('/'.join([month, day, year]))

##CONCATENAÇÃO DE STRINGS##

#uma outra forma que o python nos permite unir strings é utilizando o +

sdds = planet + ', we miss you.'
print(sdds)

#podemos usar o + pra juntar objetos a strings que não sejam strings, mas deveos trnasformá-los antes:

position = 9
coping = planet + ", you'll always be the " + str(position) + "th planet to me."

print(coping)

#porém quando maior a string mais convoluto, por isso usamos o method string.format() para acelerar o processo:

coping = "{}, you'll always be the {}th planet to me.".format(planet, position)
print(coping)

#observe como nem tivemos que transformar as informações em strings antes de lançá-las na string principal

#na verdade, isso é só umas das diversas coisas que .format() pode fazer, alguns outros exemplos são:

pluto_mass = 1.303 * 10**22
earth_mass = 5.9722 * 10**24
population = 52910390
#         2 decimal points   3 decimal points, format as percent     separate with commas
pluto_curiosidades = "{} weighs about {:.2} kilograms ({:.3%} of Earth's mass). It is home to {:,} Plutonians.".format(
    planet, pluto_mass, pluto_mass / earth_mass, population,
)
print(pluto_curiosidades)

# Referring to format() arguments by index, starting from 0
s = """Pluto's a {0}.
No, it's a {1}.
{0}!
{1}!""".format('planet', 'dwarf planet')
print(s)


###DICIONÁRIOS###

#são estruturas de dados com o objetivo de relacionar valores a chaves:

numbers = {'one':1, 'two':2, 'three':3}
print(numbers)

#as palavras são as chaves e os números os valores 

#podmeos acessar os valores num dicionários através de sua chave:

print(numbers['one'])

#também podemos usar uma sintaxe parecida para adcionar um novo par de chave,valor

numbers['eleven'] = 11
print(numbers)

#e para mudar o valor associado a uma chave

numbers['one'] = 'Pluto'
print(numbers)

#do mesmo jeito que pra litas, há uma forma comprimida de escrever dicionários:

planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
planet_to_initial = {planet: planet[0] for planet in planets}
print(planet_to_initial)

#o operador in vai nos dizer se uma chave está no dicionário

#e se looparmos em dicionários, faremos o loop sobre suas chaves

#podemos acessar uma coleção de todas as chaves e valores com dici.keys() e dici.values(), respectivamente

ordem_dos_planetas = ' '.join(sorted(planet_to_initial.values()))
# Get all the initials, sort them alphabetically, and put them in a space-separated string.

print(ordem_dos_planetas)

#e por fim, o method dici.items() permite que vc navegue pelas chaves e valores simult}aneamente

for planet, initial in planet_to_initial.items():
    print("{} begins with \"{}\"".format(planet.rjust(10), initial))