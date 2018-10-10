# import libraries
import pandas as pd
import numpy as np
import math
import pickle


# configs
# environment: gcolab - Google Colab, local - local machine
environment = 'local'
#environment= 'gcolab'
# debug mode: if it is set True, only use partial dataset for the purpose of debug or demonstration
debug_mode = True
# load_existing_w_matrix: it it is set True, the previous built similarity matrix will be loaded instead of building one
load_existing_w_matrix = True


# Set the file path where the similarity matrix will be persisted
if debug_mode == True:
    DEFAULT_PARTICLE_PATH = 'w_matrix_debug.pkl'
else:
    DEFAULT_PARTICLE_PATH = 'w_matrix.pkl'


# install libraries and authorise google drive
if environment == 'gcolab':
    get_ipython().system('apt-get install -y -qq software-properties-common python-software-properties module-init-tools')
    get_ipython().system('add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null')
    get_ipython().system('apt-get update -qq 2>&1 > /dev/null')
    get_ipython().system('apt-get -y install -qq google-drive-ocamlfuse fuse')
    from google.colab import auth
    auth.authenticate_user()
    from oauth2client.client import GoogleCredentials
    creds = GoogleCredentials.get_application_default()
    import getpass
    get_ipython().system('google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL')
    vcode = getpass.getpass()
    get_ipython().system('echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}')


# mount google drive
if environment == 'gcolab': 
    get_ipython().system('mkdir -p drive')
    get_ipython().system('google-drive-ocamlfuse drive')


# load data
if environment == 'gcolab':
    ratings = pd.read_csv("drive/Colab-Notebooks/datasets/ratings.csv", encoding='"ISO-8859-1"')
    movies = pd.read_csv("drive/Colab-Notebooks/datasets/movies.csv", encoding='"ISO-8859-1"')
    tags = pd.read_csv("drive/Colab-Notebooks/datasets/tags.csv", encoding='"ISO-8859-1"')
    DEFAULT_PARTICLE_PATH = "drive/Colab-Notebooks/" + DEFAULT_PARTICLE_PATH
else:
    ratings = pd.read_csv("datasets/ratings.csv", encoding='"ISO-8859-1"')
    movies = pd.read_csv("datasets/movies.csv", encoding='"ISO-8859-1"')
    tags = pd.read_csv("datasets/tags.csv", encoding='"ISO-8859-1"')


# use partial dataset for debug mode
if debug_mode == True:
    ratings = ratings[(ratings['movieId'] < 100) & (ratings['userId'] < 100)]
    movies = movies[movies['movieId'] < 100]


# split the ratings into training and test
ratings_training = ratings.sample(frac=0.7)
ratings_test = ratings.drop(ratings_training.index)


# calculate adjusted ratings based on training data
rating_mean= ratings_training.groupby(['movieId'], as_index = False, sort = False).mean().rename(columns = {'rating': 'rating_mean'})[['movieId','rating_mean']]
adjusted_ratings = pd.merge(ratings_training,rating_mean,on = 'movieId', how = 'left', sort = False)
adjusted_ratings['rating_adjusted']=adjusted_ratings['rating']-adjusted_ratings['rating_mean']
# replace 0 adjusted rating values to 1*e-8 in order to avoid 0 denominator
adjusted_ratings.loc[adjusted_ratings['rating_adjusted'] == 0, 'rating_adjusted'] = 1e-8


# function of building the item-to-item weight matrix
def build_w_matrix(adjusted_ratings, load_existing_w_matrix):
    # define weight matrix
    w_matrix_columns = ['movie_1', 'movie_2', 'weight']
    w_matrix=pd.DataFrame(columns=w_matrix_columns)

    # load weight matrix from pickle file
    if load_existing_w_matrix:
        with open(DEFAULT_PARTICLE_PATH, 'rb') as input:
            w_matrix = pickle.load(input)
        input.close()

    # calculate the similarity values
    else:
        distinct_movies = np.unique(adjusted_ratings['movieId'])

        i = 0
        # for each movie_1 in all movies
        for movie_1 in distinct_movies:

            if i%10==0:
                print(i , "out of ", len(distinct_movies))

            # extract all users who rated movie_1
            user_data = adjusted_ratings[adjusted_ratings['movieId'] == movie_1]
            distinct_users = np.unique(user_data['userId'])

            # record the ratings for users who rated both movie_1 and movie_2
            record_row_columns = ['userId', 'movie_1', 'movie_2', 'rating_adjusted_1', 'rating_adjusted_2']
            record_movie_1_2 = pd.DataFrame(columns=record_row_columns)
            # for each customer C who rated movie_1
            for c_userid in distinct_users:
                print('build weight matrix for customer %d, movie_1 %d' % (c_userid, movie_1))
                # the customer's rating for movie_1
                c_movie_1_rating = user_data[user_data['userId'] == c_userid]['rating_adjusted'].iloc[0]
                # extract movies rated by the customer excluding movie_1
                c_user_data = adjusted_ratings[(adjusted_ratings['userId'] == c_userid) & (adjusted_ratings['movieId'] != movie_1)]
                c_distinct_movies = np.unique(c_user_data['movieId'])

                # for each movie rated by customer C as movie=2
                for movie_2 in c_distinct_movies:
                    # the customer's rating for movie_2
                    c_movie_2_rating = c_user_data[c_user_data['movieId'] == movie_2]['rating_adjusted'].iloc[0]
                    record_row = pd.Series([c_userid, movie_1, movie_2, c_movie_1_rating, c_movie_2_rating], index=record_row_columns)
                    record_movie_1_2 = record_movie_1_2.append(record_row, ignore_index=True)

            # calculate the similarity values between movie_1 and the above recorded movies
            distinct_movie_2 = np.unique(record_movie_1_2['movie_2'])
            # for each movie 2
            for movie_2 in distinct_movie_2:
                print('calculate weight movie_1 %d, movie_2 %d' % (movie_1, movie_2))
                paired_movie_1_2 = record_movie_1_2[record_movie_1_2['movie_2'] == movie_2]
                sim_value_numerator = (paired_movie_1_2['rating_adjusted_1'] * paired_movie_1_2['rating_adjusted_2']).sum()
                sim_value_denominator = np.sqrt(np.square(paired_movie_1_2['rating_adjusted_1']).sum()) * np.sqrt(np.square(paired_movie_1_2['rating_adjusted_2']).sum())
                sim_value_denominator = sim_value_denominator if sim_value_denominator != 0 else 1e-8
                sim_value = sim_value_numerator / sim_value_denominator
                w_matrix = w_matrix.append(pd.Series([movie_1, movie_2, sim_value], index=w_matrix_columns), ignore_index=True)

            i = i + 1

        # output weight matrix to pickle file
        with open(DEFAULT_PARTICLE_PATH, 'wb') as output:
            pickle.dump(w_matrix, output, pickle.HIGHEST_PROTOCOL)
        output.close()

    return w_matrix

# run the function to build similarity matrix
w_matrix = build_w_matrix(adjusted_ratings, load_existing_w_matrix)

# calculate the predicted ratings
def predict(userId, movieId, w_matrix, adjusted_ratings, rating_mean):
    # fix missing mean rating which was caused by no ratings for the given movie
    # mean_rating exists for movieId
    if rating_mean[rating_mean['movieId'] == movieId].shape[0] > 0:
        mean_rating = rating_mean[rating_mean['movieId'] == movieId]['rating_mean'].iloc[0]
    # mean_rating does not exist for movieId(which may be caused by no ratings for the movie)
    else:
        mean_rating = 2.5

    # calculate the rating of the given movie by the given user
    user_other_ratings = adjusted_ratings[adjusted_ratings['userId'] == userId]
    user_distinct_movies = np.unique(user_other_ratings['movieId'])
    sum_weighted_other_ratings = 0
    sum_weghts = 0
    for movie_j in user_distinct_movies:
        if rating_mean[rating_mean['movieId'] == movie_j].shape[0] > 0:
            rating_mean_j = rating_mean[rating_mean['movieId'] == movie_j]['rating_mean'].iloc[0]
        else:
            rating_mean_j = 2.5
        # only calculate the weighted values when the weight between movie_1 and movie_2 exists in weight matrix
        w_movie_1_2 = w_matrix[(w_matrix['movie_1'] == movieId) & (w_matrix['movie_2'] == movie_j)]
        if w_movie_1_2.shape[0] > 0:
            user_rating_j = user_other_ratings[user_other_ratings['movieId']==movie_j]
            sum_weighted_other_ratings += (user_rating_j['rating'].iloc[0] - rating_mean_j) * w_movie_1_2['weight'].iloc[0]
            sum_weghts += np.abs(w_movie_1_2['weight'].iloc[0])

    # if sum_weights is 0 (which may be because of no ratings from new users), use the mean ratings
    if sum_weghts == 0:
        predicted_rating = mean_rating
    # sum_weights is bigger than 0
    else:
        predicted_rating = mean_rating + sum_weighted_other_ratings/sum_weghts

    return predicted_rating

# predict a rating for a given user and given movie
predicted_rating = predict(2, 29, w_matrix, adjusted_ratings, rating_mean)
print('The predicted rating: %f' % predicted_rating)

# evaluate the learned recommender system on test data by converting the ratings to negative and positive
def binary_eval(ratings_test, w_matrix, adjusted_ratings, rating_mean):
    # predict all the ratings for test data
    ratings_test = ratings_test.assign(predicted_rating = pd.Series(np.zeros(ratings_test.shape[0])))
    for index, row_rating in ratings_test.iterrows():
        predicted_rating = predict(row_rating['userId'], row_rating['movieId'], w_matrix, adjusted_ratings, rating_mean)
        ratings_test.loc[index, 'predicted_rating'] = predicted_rating
    tp = ratings_test.query('(rating >= 2.5) & (predicted_rating >= 2.5)').shape[0]
    fp = ratings_test.query('(rating < 2.5) & (predicted_rating >= 2.5)').shape[0]
    fn = ratings_test.query('(rating >= 2.5) & (predicted_rating < 2.5)').shape[0]

    # calculate the precision and recall
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    return (precision, recall)

# run the evaluation
eval_result = binary_eval(ratings_test, w_matrix, adjusted_ratings, rating_mean)
print('Evaluation result - precision: %f, recall: %f' % eval_result)

# make recommendations
def recommend(userID, w_matrix, adjusted_ratings, rating_mean, amount=10):
    distinct_movies = np.unique(adjusted_ratings['movieId'])
    user_ratings_all_movies = pd.DataFrame(columns=['movieId', 'rating'])
    user_rating = adjusted_ratings[adjusted_ratings['userId']==userID]

    # calculate the ratings for all movies that the user hasn't rated
    i = 0
    for movie in distinct_movies:
        user_rating = user_rating[user_rating['movieId']==movie]
        if user_rating.shape[0] > 0:
            rating_value = user_ratings_all_movies.loc[i, 'rating'] = user_rating.loc[0, movie]
        else:
            rating_value = user_ratings_all_movies.loc[i, 'rating'] = predict(userID, movie, w_matrix, adjusted_ratings, rating_mean)
        user_ratings_all_movies.loc[i] = [movie, rating_value]

        i = i + 1

    # select top 10 movies rated by the user
    recommendations = user_ratings_all_movies.sort_values(by=['rating'], ascending=False).head(amount)
    return recommendations

# get a recommendation list for a given user
recommended_movies = recommend(2, w_matrix, adjusted_ratings, rating_mean)
print(recommended_movies)
