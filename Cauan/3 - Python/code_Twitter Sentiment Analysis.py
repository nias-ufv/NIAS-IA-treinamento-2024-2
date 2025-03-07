###Twitter Sentiment Analysis###

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
"Freecodecamp has great coding tutorials. #skillup"]

##QUESTÃO 1##

print("1. Quantos tweets existem nesse banco de dados? {} tweets".format(len(tweets)))

##QUESTÃO 2##

happy_words = ['great', 'excited', 'happy', 'nice', 'wonderful', 'amazing', 'good', 'best']
sad_words = ['sad', 'bad', 'tragic', 'unhappy', 'worst']

def is_tweet(tweet, mood):

    words = tweets[tweet-1].split()
    ishappy = issad = nomood = False
    
    for word in words:
        clean_word = word.lower().strip('.,!#')
        if clean_word in happy_words:
            ishappy = True
        elif clean_word in sad_words:
            issad = True
    
    return (ishappy and mood=='happy') or (issad and mood=='sad')

print("O tweet número {} é um tweet {}!".format(1, "feliz" if is_tweet(1, 'happy')==True else "triste ou sem um sentimento específico"))

#Pra fins de fácil interpretação, os tweets estão sendo contados como se a indexação começasse em 1
#Ex.: tweet número 1 = tweets[0]

##QUESTÃO 3##

happy_twts = sum([is_tweet(n+1, 'happy') for n in range(len(tweets))])
print("No total, {} dos tweets são considerados felizes :)".format(happy_twts))

sad_twts = sum([is_tweet(n+1, 'sad') for n in range(len(tweets))])
print("No total, {} dos tweets são considerados tristes :(".format(sad_twts))

norm_tweets = abs(sad_twts + happy_twts - len(tweets))
print("No total, {} dos tweets não possuem um sentimento identificável :|".format(norm_tweets))