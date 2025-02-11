spam_amount = 0
#isso é uma associação de valores, os python associa valores a variaveis a partir do sinal de = uma vez.
#voce não precisa declarar o tipo de valor

print(spam_amount)
#essa é a função print. funções são coisas que você pode fazer no circuito, algumas são próprias do python
#outras você pode criar por si mesmo. chamamos funções escrevendo seu nome e um par de parênteses a frente
#delas. Dentro desses parênteses, colocamos os chamados parâmetros.

# Ordering Spam, egg, Spam, Spam, bacon and Spam (4 more servings of Spam)
#isso é um comentário, não são lidos no ódgio

spam_amount = spam_amount + 4
#aqui tem uma reassociação. Ele basicamente faz uma conta matemática do valor anterior da variável somado a um inteiro

if spam_amount > 0:
    print("But I don't want ANY spam!")
#uso de um condicional. 'if' indica uma situação, indentada logo abaixo, que deve acontecer apenas se outra situação acontecer primeiro,
#ou se alguma coisa estiver no lugar que deveria, ou se alguma coisa tem o valor que deveria, etc.

viking_song = "Spam " * spam_amount
#Spam aqui surge como uma string, indicada pelas aspas. Observe que, mesmo com uma string, é possível fazeer operações matemáticas
print(viking_song)

#final do código de exemplo


print(type(spam_amount))
print(type(viking_song))
print(type(19.5))
#testando a função type, que dá o tipo de variável que cada coisa representa

#exemplos de funções matemáticas:
a = 2
b = 4

print(a)
print(b)

print(a + b)
#soma

print(a - b)
#subtração

print(a*b)
#multiplicação

print(a/b)
#divisão com resultado float

print(a//b)
#divisão só com resultado integer

print(a%b)
#módulo da subtração

print(a**b)
#a elevado a b

print(-a)
#negativização


#(algumas) funções pré montadas pra trabalhar com números:

print(min(a,b))
print(max(a,b))
#min/max: dão o valor máximo entre os números fornecidos

#teste:
nmr=[1,2,3,4]
print(max(nmr))
print(min(nmr))
#note que essas funções recebem valores mais complexos, como listas

print(abs(-2))
#abs: retorna o módulpo de um valor

print(int(10.6))
print(float(10))
print(int('10')+5)
#int e float: além de serem tipos de valores, podem ser usado como função para redefinir um valor dado. isso serve até mesmo para strings