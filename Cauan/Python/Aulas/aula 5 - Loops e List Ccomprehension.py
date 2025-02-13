###LOOPS###

#são formas de repetir um mesmo código várias vezes

##FOR##

planetas = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']

for planeta in planetas:
    print(planeta,end=' ')
    
print(end='\n')
    
#o for tem como 'parametros' uma variável, criada na hora,
#usada como referência pra navegação dentro de um elemento
#que permita indexação, que é o outro parâmetro do for

#note que isso inclui os óbivos, como listas e tuplas, mas
#coisas como strings:

s = 'steganograpHy is the practicE of conceaLing a file, message, image, or video within another fiLe, message, image, Or video.'
msg = ''
# print all the uppercase letters in s, one at a time
for char in s:
    if char.isupper():
        print(char, end='')
        
print(end='\n')

#uma função comumente utilizada em conjunção com o for é
#a range(), que entrega uma sequência de números.

for i in range(5):
    print("Doing important work. i =", i)
    
##WHILE##

#while é o outro tipo de loop no python. ele itera
#até que uma dada condição ocorra

i = 0
while i < 10:
    print(i, end=' ')
    i += 1 # increase the value of i by 1
    
print(end='\n')
    
##LIST COMPREHENSION##

#basicamente um jeito de criar uma lista baseada em
#um loop ao inves de um set predefinido de números:

squares = [n**2 for n in range(10)]
print(squares)

#podemos até extrapolar, adcionando um if no loop

pqns_planetas = [planeta for planeta in planetas if len(planeta) < 6]
print(pqns_planetas)

#fez uma lista só de planetas com nomes com menos de 6 letras

#podemos adcionar masi um nivel de complexidade adcionando
#também transformações na criação da lista

gritar_pqns_planetas = [planeta.upper() + '!' for planeta in planetas if len(planeta) < 6]
print(gritar_pqns_planetas)

###FIM DA AULA###