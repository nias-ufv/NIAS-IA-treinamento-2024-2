####FUNÇÕES####

help(round)
#função que recebe como parâmetro uma função qualquer do python e descreve sua funcionalidade, além do que ela recebe como parâmetro

#note que, se você chamar help numa função já dando seus parâmetros, ela vai falar sobre o resultado da função
#help(round(2.01)), dei comentário pq tava saindo um output mt grande e bugando o terminal

#help brilha ao falar sobre funções mais complexas, por exemplo com várias opções de parametros. por exemplo, o print
help(print)

#teste:
print(1,2,3,sep=' e outro número ',end='\n cauan \n')

#além das funções que o próprio python possui, podemos criar nossas próprias, atraves da definição de novas funções, por exemplo:

def menor_diferença(a,b,c):
    dif1 = abs(a-b)
    dif2 = abs(b-c)
    dif3 = abs(c-a)
    return min(dif1,dif2,dif3)

print(menor_diferença(1,2,3))

#observe oq acontece ao chamarmos o help para uma função que definimos:
help(menor_diferença)
#ele não sabe exatamente o que tem que ser feito, mas podemos dizer pra ele o que realmente acontece através de um texto na função.
#esse texto é chamado "docstring". Observe como podemos fazer isso:

def maior_diferença(a,b,c):
    """Retorna a maior diferença entre qualquer subtração contendo a, b e c (sem subtraão entre iguais)
    
    >>> maior_diferença(1, 5, -5)
    10"""
    dif1=abs(a-b)
    dif2=abs(b-c)
    dif3=abs(c-a)

    return max(dif1,dif2,dif3)

print(maior_diferença(1,5,-5))

help(maior_diferença)


##PARÂMETRO PADRÃO##

#Uma outra característica de funções que podemos utilizar é preparar um parâmetro com um padrão pré estabelecido. 
#Ou seja, se caso a função for chamada sem qualquer parâmetro, ela assume seu parâmetro padrão. Em qualquer outro caso, ela substitui
#o parâmetro padrão pelo informado. Exemplo:

def opabão(quem="Carlos"):
    """cumprimenta um cumpadi. Se não lembrar o nome, cumprimenta o Carlos"""
    print("opa bão,", quem)

opabão("João")
opabão(quem="Márcio")
opabão()


##FUNCEPTION, USAR FUNÇÕES COMO ARGUMENTO DE FUNÇÕES##

#Uma outra ideia que podemos usar é usar os resultados de funções para alimentar otras funções. Pudemos ver um uso disso ao chamar
#a função help() em uma função com argumentos, mas tornar isso ainda mais interessante, apesar de um pouco convoluto. Observe:

def multpor2(a):
    rstd=a*2
    return(rstd)

def chamar(fç, prmt):
    """chama uma função já passando um parâmetro"""
    return fç(prmt)

def chamardenovo(fç, prmt):
    """chama uma função duas vezes sobre um mesmo parâmetro"""
    return fç(fç(prmt))

print(chamar(multpor2,1))
print(chamardenovo(multpor2,1))

#essas funções são chamadas "higher order functions", ou funções de ordem superior

###FIM DA AULA###