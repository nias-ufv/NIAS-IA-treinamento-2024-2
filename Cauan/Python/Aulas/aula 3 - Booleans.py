###VARIÁVEIS BOOLEANAS###

#No Python, as variáveis chamadas booleanas são as que só possuem dois valores: true ou false. Exemplo:

x = True
print(x)
print(type(x))

#esses são os operadores que respondem perguntas de sim ou não

##OPERAÇÕES D ECOMPARAÇÃO##

#são elas:

#a==b, a é igual a b
#a<b, a é menor que b
#a<=b, a é menor ou igual a b
#a!=b, a não é igual a b
#a>b, a é maior que b
#a>=b, a é maior ou igual a b

#exemplo de uso:

def pode_se_candidatar(idd):
    """Diz se uma pessoa com a idade dada como parâmetro
    pode se candidatar a presidente"""
    
    return idd>=35

print(pode_se_candidatar(19))
print(pode_se_candidatar(49))

#tome cuidado ao comparar variáveis de com classes mt diferentes
#um int com float? Ok. Um float com string? Não tão ok.

#comparações podem nos ajudar fazendo comparações matemáticas
#muito úteis. Por exemplo:

def é_par(num):
    return (num%2) == 0

print(é_par(100))
print(é_par(101))

##COMBINAÇÕES DE VALORES BOOLEANOS##    

#você pode combinar valores booleanos utilizando as palavras
#and, or ou not. Observe:

def pode_concorrer(idd,é_cidadão):
    """Alguém com a idade e o estado de cidadania pode
    concorrer a presidência?
    
    Note que, a Constituição Brasileira diz que para
    concorrer, a pessoa deve ser natural do Brasil ""e""
    ter pelo menos 25"""
    
    return é_cidadão and (idd>=35)

print(pode_concorrer(45,False))
print(pode_concorrer(45,True))

#ao usar comparações boolenas, existe uma ordem de 
#prioridade, como a das continhas, mas é mais prático
#sempre usar parentêses

##CONDICIONAIS##

#são os famosos if, elif e else. Eles permitem controlar
#que partes do código rodam baseados no valor de uma
#condição Booleana. Por exemplo:

def inspecione(n):
    if n==0:
        print("n é zero otário")
    elif n>0:
        print("n é maior que zero otário")
    elif n<0:
        print("n é menor que 0, bestoide")
    else:
        print("n é o tipo de questão que cai na prova do edinho")

inspecione(0)
inspecione(10)

#elif = else if. Note que, para um mesmo if, podemos incluir
#quantos elif quisermos, mas apenas um else.


##CONVERSÃO BOOLEANA##

#O Python possui uma função de conversão de qualquer
#coisa para booleano: bool(). De forma geral, tudo que for
#informado como parâmetro que possuir um valor, é considerado
#true. Se está vazio ou o valor é 0, é considerado False.
#Podemos usar essefato em condicionais:

if 0:
    print(0)
elif "opabão":
    print("opabão")



###FIM DA AULA###