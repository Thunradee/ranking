import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import stop_words
from nltk.tokenize import TreebankWordTokenizer
import numpy as np
import time

# dictionaries to hold data read from files
movie_title = {}                # movie titles by movieId
movie_year = {}                 # movie year by movieId
movie_genres = {}               # list of genre keywords, by movieId
movie_plot = {}                 # movie plots by movieId
movie_imdb_rating = {}          # movie IMDb rating by movieId
user_ratings = {}               # list of (movieId, rating, timestamp) by userID

def read_data():
    '''
    Reads in data from files
    '''
    global movie_title, movie_year, movie_genres, movie_plot, movie_imdb_rating, user_ratings
    # read movie titles, years, and genres
    #with open('/content/gdrive/My Drive/ML/HW/movies.csv') as csv_file:
    with open('ml-latest-small/movies.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        line_num = 0
        for row in csv_reader:
            if line_num > 0: # skip header
                movieId = int(row[0])
                title_year = row[1]
                if title_year[0] == '"':
                    title_year = title_year[1:-1]
                genres = row[2]
                title_year = title_year.strip()
                movie_title[movieId] = title_year[:-7]
                if title_year[-1] == ')':
                    movie_year[movieId] = int(title_year[-5:-1])
                else:
                    movie_year[movieId] = 0
                if genres == "(no genres listed)":
                    movie_genres[movieId] = []
                else:
                    movie_genres[movieId] = genres.split('|')
            line_num += 1
    # read movie plots
    #with open('/content/gdrive/My Drive/ML/HW/plots-imdb.csv') as csv_file:
    with open('ml-latest-small/plots-imdb.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        line_num = 0
        for row in csv_reader:
            if line_num > 0: # skip header
                movieId = int(row[0])
                plot = row[1]
                movie_plot[movieId] = plot
            line_num += 1
    # read movie IMDb ratings
    #with open('/content/gdrive/My Drive/ML/HW/ratings-imdb.csv') as csv_file:
    with open('ml-latest-small/ratings-imdb.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        line_num = 0
        for row in csv_reader:
            if line_num > 0: # skip header
                movieId = int(row[0])
                rating = float(row[1])
                movie_imdb_rating[movieId] = rating
            line_num += 1
    # read user ratings of movies
    #with open('/content/gdrive/My Drive/ML/HW/ratings.csv') as csv_file:
    with open('ml-latest-small/ratings.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        line_num = 0
        for row in csv_reader:
            if line_num > 0: # skip header
                userId = int(row[0])
                movieId = int(row[1])
                rating = float(row[2])
                timestamp = int(row[3])
                # store user ratings in a dictionary by userID
                user_rating = (movieId, rating, timestamp)
                if userId in user_ratings:
                    user_ratings[userId].append(user_rating)
                else:
                    user_ratings[userId] = [user_rating]
            line_num += 1

def buildDataSet(movie_num_limit=0):
    '''
    Create data set
    :param movie_num_limit: number of movies that will be included into data set from each user. (movie_num_limit=0) = no limit
    :return: a data set for training and testing
    '''
    global user_ratings

    movie_ids = []

    for userID, ratings in user_ratings.items(): # for every user
        i = 0
        if movie_num_limit == 0:
            for movieID, rating, ts in ratings:
                if movieID not in movie_ids:
                    movie_ids.append(movieID)
        else:
            for movieID, rating, ts in ratings:
                if i >= movie_num_limit:
                    break
                if movieID not in movie_ids:
                    movie_ids.append(movieID)
                i += 1

    return movie_ids

def bagOfWords(movie_ids, stopWords, wordFreq):
    '''
    Creates a bag of words
    :param movie_ids: A list of movie ids
    :param stopWords: A boolean determine if want to use stop word technique
    :param wordFreq: A boolean determine if want to use word frequency inverse technique
    :return: DTM of word count
    '''
    global movie_title, movie_genres, movie_plot

    # convert collection of documents to matrix of string (word) counts
    count_vect = CountVectorizer()

    # use regular expressions to convert text to tokens
    # split contractions, separate punctuation
    tokenizer = TreebankWordTokenizer()
    count_vect.set_params(tokenizer=tokenizer.tokenize)

    if stopWords == True:
        # remove English stop words
        count_vect.set_params(stop_words='english')

    # include 1-grams and 2-grams
    count_vect.set_params(ngram_range=(1, 2))

    # ignore terms that appear in >50% of the documents
    count_vect.set_params(max_df=0.5)

    # ignore terms that appear in only 1% document
    count_vect.set_params(min_df=0.1)

    # make a list of words
    words = list()
    for m in movie_ids:
        line = ' '.join(movie_genres.get(m, [])) + ' ' + movie_title.get(m, '') + ' ' + movie_plot.get(m, '')
        words.append(line)

    # transform text to bag of words vector using parameters
    word_count = count_vect.fit_transform(words)

    if wordFreq == True:
        # normalize counts based on document length
        # weight common words less (is, a, an, the)
        tfidf_transformer = TfidfTransformer()
        word_count = tfidf_transformer.fit_transform(word_count)

    return word_count

def toFeatures(movie_ids, word_count):
    '''
    Builds a movie's features list
    :param movie_ids: A list of movie ids
    :param word_count: A DTM of word count
    :return: a list of movie's features
    '''
    global movie_year

    # convert DTM object to an array
    wc = word_count.toarray()

    features = []
    for i, m in enumerate(movie_ids):
        f = [movie_year[m]] + wc[i].tolist()
        features.append(f)

    return features

def userPreference(m1_rating, m2_rating):
    '''
    Determine if movie 1 is preferred to movie 2
    :param m1_rating: Rating of movie 1
    :param m2_rating: Rating of movie 2
    :return: 1 if movie 1 is preferred to movie 2, 0 otherwise
    '''
    preferred = 0
    if m1_rating > m2_rating:
        preferred = 1

    return preferred

def RankTrain(features_dict, classifier, movie_num_limit=0):
    '''
    Trains a classifier for ranking
    :param features_dict: A dictionary of movie's features [year, --word count-- ] by movieID
    :param classifier: A classifier that will be trained
    :param movie_num_limit: A limit number of movie of each user
    :return: A trained ranking classifier
    '''
    print("Building Training Set ...")
    start_time = time.time()
    # build training set
    X = [] # features of two movies
    y = [] # is movie1 is preferred to movie2, 1 if preferred, 0 otherwise

    # for all users, we look at the first movie_num_limit movies
    for userID, ratings in user_ratings.items():
        #print("userID:", userID)

        # determine limit (some users rated movies less than movie_num_limit)
        limit = min(movie_num_limit, len(ratings))

        # build data point by look at two movies at a time
        # (m[1], m[2]), (m[1], m[3]), ..., (m[2], m[3]), (m[2], m[4]), ..., (m[n-1], m[n])
        for i in range(limit - 1):
            m1ID = ratings[i][0]
            #print("m1ID", m1ID)
            for j in range(i + 1, limit):
                m2ID = ratings[j][0]
                #print("m2ID", m2ID)

                # determine preference
                preferred = userPreference(ratings[i][1], ratings[j][1])
                #print('Preferred:', preferred, "(", ratings[i][1], ratings[j][1],")")
                y.append(preferred)
                xij = features_dict[m1ID] + features_dict[m2ID]
                X.append(xij)
    print("Training Set lenght:", len(y))
    print("--- %s seconds ---" % (time.time() - start_time))

    # train a classifier
    print()
    print("Training Classifier ...")
    start_time = time.time()
    clf = classifier.fit(X, y)
    print("--- %s seconds ---" % (time.time() - start_time))

    return clf

def predict(clf, features_dict, m1ID, m2ID):
    '''
    Calculate probability that movie m1ID will have higher rank than movie m2ID
    :param clf: A trained classifier
    :param features_dict: A dictionary of movie's features [year, --word count-- ] by movieID
    :param m1ID: First movie id
    :param m2ID: Second movie id
    :return: probability that movie m1ID will have higher rank than movie m2ID
    '''
    m1x = features_dict[m1ID]
    m2x = features_dict[m2ID]
    x = m1x + m2x
    #prob = clf.predict_proba([x])
    #prob = prob[0][0]
    preferred = clf.predict([x])

    return preferred

def partition(clf, features_dict, movie_ids, l, r):
    '''
    Partitions movies (Inplace)
    :param clf: A trained classifier
    :param features_dict: A dictionary of movie's features [year, --word count-- ] by movieID
    :param movie_ids: A list of movie ids that will be ranked
    :param l: Left index
    :param r: Right index
    :return: The pivot index
    '''
    i = l - 1
    p = movie_ids[r] # pivot ID

    for j in range(l, r):
        u = movie_ids[j] # movie u ID
        '''
        prob = predict(clf, features_dict, u, p)
        if prob > 0.5:
            i += 1
            movie_ids[i], movie_ids[j] = movie_ids[j], movie_ids[i]
        else:
            print('prob:', prob)
        '''
        preferred = predict(clf, features_dict, u, p)
        if preferred == 1:
            i += 1
            movie_ids[i], movie_ids[j] = movie_ids[j], movie_ids[i]
        #else:
            #print('mID:', u)

    movie_ids[i+1], movie_ids[r] = movie_ids[r], movie_ids[i+1]

    #print(i)
    return i + 1

def RankTest(clf, features_dict, movie_ids, l, r):
    '''
    Ranks movies (Inplace)
    :param clf: A trained classifier
    :param features_dict: A dictionary of movie's features [year, --word count-- ] by movieID
    :param movie_ids: A list of movie ids that will be ranked
    :param l: Left index
    :param r: Right index
    '''
    # create stack
    size = r - l + 1
    stack = [0] * size
    top = -1

    # push initial l and r to the stack
    top += 1
    stack[top] = l
    top += 1
    stack[top] = r

    while top >= 0:
        # Pop r and l
        r = stack[top]
        top -= 1
        l = stack[top]
        top -= 1

        # Set pivot element at its correct position in
        # sorted array
        p = partition(clf, features_dict, movie_ids, l, r)


        # If there are elements on left side of pivot,
        # then push left side to stack
        if p - 1 > l:
            top += 1
            stack[top] = l
            top += 1
            stack[top] = p - 1

        # If there are elements on right side of pivot,
        # then push right side to stack
        if p + 1 < r:
            top += 1
            stack[top] = p + 1
            top += 1
            stack[top] = r



def getIMDBRank(movie_ids):
    '''
    Create IMBD ranking base on IMBD rating
    :param movie_ids: A list of movie ids that will be ranked
    :return: IMBD ranking
    '''
    # Merge Sort
    global movie_imdb_rating

    if len(movie_ids) <= 1:
        return movie_ids

    # pivot is the last element of the movie_ids list
    p_rating = -1
    while p_rating == -1:
        p = movie_ids[-1]
        # return value of the key p, if p in movie_imdb_rating
        # otherwise return -1
        p_rating = movie_imdb_rating.get(p, -1)

        # don't include movie that doesn't IMBD rating in IMBD ranking
        if p_rating == -1:
            movie_ids = movie_ids[:-1]

    # partition higher rank movies to the left of the pivot movie, and lower rank movies to the right
    left = []
    right = []
    for mID in movie_ids[:-1]:
        m_rating = movie_imdb_rating.get(mID, -1)  # get rating of the movie, return -1 if the movie is not in movie_imdb_rating
        if m_rating == -1:
            pass  # don't include movie that doesn't IMBD rating in IMBD ranking

        # put the movie to the left if the movie has higher rating than the pivot movie
        elif m_rating > p_rating:
            left.append(mID)

        # put the movie to the right if the movie has lower rating than the pivot movie
        # assume that no ties
        else:
            right.append(mID)

    # recursive call on the left
    left = getIMDBRank(left)
    # recursive call on the right
    right = getIMDBRank(right)

    return left + [p] + right

def avgKemenyDistance(predRank, baseRank):
    # focus on top 50
    k = 0
    j = 0
    for i in range(50):
        mID = predRank[i]
        if mID in baseRank:
            k += abs(i-baseRank.index(mID))
            j += 1
        else:
            print("IMBD doesn't have:", mID)

    return k/j

if __name__ == "__main__":
    #global movie_title, ranking_limit
    #print("Reading data...", flush=True)
    read_data()
    limit = 10

    # build data set
    print("Building Data Set ...")
    movie_ids = buildDataSet(limit)
    movie_num = len(movie_ids)
    print("Dataset")
    print(movie_ids)
    print('movie_ids length =', len(movie_ids))

    # create bag of words
    print("Word Engineering ...")
    start_time = time.time()
    wc = bagOfWords(movie_ids, stopWords=False, wordFreq=True)
    print("--- %s seconds ---" % (time.time() - start_time))

    # add year in to the front of word count
    features = toFeatures(movie_ids, wc)

    # a dictionary of feature vector by movieID
    features_dict = dict(zip(movie_ids, features))

    # train ranking classifier
    #classifier = KNeighborsClassifier(n_neighbors=5)
    classifier = MultinomialNB()
    clf = RankTrain(features_dict, classifier, limit)

    # predict ranking
    print("Ranking ...")
    start_time = time.time()
    rank = movie_ids.copy()
    RankTest(clf, features_dict, rank, 0, movie_num-1)
    print("--- %s seconds ---" % (time.time() - start_time))

    print("Dataset")
    print(movie_ids)

    # rank movie base on IMBD rating
    imbdRank = getIMDBRank(movie_ids)
    print("User Rank")
    print(rank)
    print("IMBD Rank")
    print(imbdRank)

    # claculate Kemeny Distance
    avgKemeny = avgKemenyDistance(rank, imbdRank)

    print("Average Kemeny (Top 50):", avgKemeny)