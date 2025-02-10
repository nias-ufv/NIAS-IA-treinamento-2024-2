# Estudos em python


tweets = [ 
"Wow, what a great day today!! #sunshine",
"I feel sad about the things going on around us. #covid19",
"I'm really excited to learn Python with @JovianML #zerotopandas",
"This is a really nice song. #linkinpark",
"The python programming language is useful for data science",
"Why do bad things happen to me?",
"Apple announces the release of the new iPhone 12. Fans are excited.",
"Spent my day with family!! #happy",
"Check out my blog post on common string operations in Python. #zerotopandas",
"Freecodecamp has great coding tutorials. #skillup" 
        ]

# Classificar os tweets em positivos, negativos e neutros
positivos = [ "great", "excited", "nice", "useful", "happy" ]
negativos = [ "sad", "bad", "tragedy", "unhappy" ]

# Contagem de tweets positivos, negativos e neutros
positivos_count = 0
negativos_count = 0
neutros_count = 0

happy = [" ", " ", " ", " ", " ", " ", " ", " ", " ", " "]
unhappy = [" ", " ", " ", " ", " ", " ", " ", " ", " ", " "]

# Classificar os tweets
for i in range(0, len(tweets)):
    tweet = tweets[i]
    words = tweet.split(' ')
    for word in words:
        if word in positivos:
            positivos_count += 1
            happy[i] = tweets[i]
            break

        elif word in negativos:
            negativos_count += 1
            unhappy[i] = tweets[i]
            break
        
    else:
        neutros_count += 1
    
    happiness = False



# Resultados
for i in range(0, len(happy)):
    if happy[i] != " ":
        print("Positivos:", happy[i])

for i in range(0, len(unhappy)):
    if unhappy[i] != " ":
        print("Negativos:", unhappy[i])

print("Positivos:", positivos_count, "Negativos:", negativos_count, "Neutros:", neutros_count)
