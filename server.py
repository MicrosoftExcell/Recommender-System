import numpy as np
import pandas as pd
import sys
import random, json
from scipy.sparse.linalg import svds
from flask import Flask, render_template,request,redirect,Response,url_for

import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

user_id = "1"
num = 10

#read in user and book data
#dataset adapted from https://github.com/zygmuntz/goodbooks-10k
ratings_data = pd.read_csv("ratings.csv")
book_data = pd.read_csv("book_data.csv")
ratings_data = ratings_data.astype({'user_id':'str'})
all_data = pd.merge(ratings_data, book_data, on='book_id', how='inner')

#getting the average ratings and number of ratings
highest_rated_data = all_data.groupby('title')['rating'].mean().sort_values(ascending=False)
most_rated_data = all_data.groupby('title')['rating'].count().sort_values(ascending=False)
all_data = pd.merge(all_data, highest_rated_data, on='title', how='left')
all_data = pd.merge(all_data, most_rated_data, on='title', how='left')
all_data = all_data.rename(columns={"rating":"num_ratings","rating_y":"avg_rating","rating_x":"rating"})

users = np.array(all_data['user_id'])

def overall_best(n):
    global best_books
    #get the most rated books for users who have no ratings yet
    #choosing highest rated out of books with top 10% num_ratings
    most_rated_books = all_data.num_ratings.quantile([.9])
    best_books = all_data[all_data['num_ratings']>most_rated_books[0.9]]
    best_books = best_books.groupby('title',as_index=False).agg({"rating":"mean"}).sort_values(by=['rating'],ascending=False).head(n)
    best_books = best_books.round({'rating':2})
    best_books = np.array(best_books)
    return best_books


# using portions of code from https://beckernick.github.io/matrix-factorization-recommender/
def matrix_factorization(full_ratings,n):
    global best_books
    user_book_rating = all_data.pivot_table(index='user_id',columns = 'title',values='rating')
    user_books = user_book_rating.fillna(0).as_matrix()
    user_mean = np.mean(user_books,axis=1)
    user_books_normal = user_books-user_mean.reshape(-1,1)
    U,sigma,Vt = svds(user_books_normal,k=50)
    sigma = np.diag(sigma)
    predicted_ratings = np.dot(np.dot(U,sigma),Vt)+user_mean.reshape(-1,1)
    predictions = pd.DataFrame(predicted_ratings,columns = user_book_rating.columns)
    predictions['user_id'] = user_book_rating.index
    user_predictions = predictions[predictions['user_id'] == user_id]
    user_predictions = user_predictions.drop(['user_id'],axis=1)
    user_predictions = user_predictions.sort_values(by=user_predictions.index[0],ascending = False,axis=1)
    recommendations = (book_data[~book_data['book_id'].isin(full_ratings['book_id'])]
                       .merge(pd.DataFrame(user_predictions).transpose(),how='left',left_on='title',right_on='title')
                       .rename(columns = {user_predictions.index[0]:'Predictions'})
                       .sort_values('Predictions',ascending = False))
    recommendations = pd.merge(recommendations,most_rated_data, on='title',how='left')
    most_rated_books = recommendations.rating.quantile([.67])
    recommendations = recommendations[recommendations['rating']>most_rated_books[0.67]]
    recommendations = recommendations.head(n)
    avg_ratings = []
    for i in range(n):
        book_avg = all_data[all_data['book_id'] == recommendations['book_id'].iloc[i]]
        avg_value = book_avg['avg_rating'].iloc[0]
        avg_ratings.append(avg_value)
    avg_ratings = [round(i,2) for i in avg_ratings]
    recommendations['avg_ratings'] = avg_ratings
    recommendations = np.array(recommendations[['title','avg_ratings']])
    return recommendations 

# get current state of dataset
def reinitialize():
    global ratings_data
    global book_data
    global all_data
    global best_books
    #read in user and book data
    ratings_data = pd.read_csv("ratings.csv")
    book_data = pd.read_csv("book_data.csv")
    ratings_data = ratings_data.astype({'user_id':'str'})
    all_data = pd.merge(ratings_data, book_data, on='book_id', how='inner')

    #getting the average ratings and number of ratings
    highest_rated_data = all_data.groupby('title')['rating'].mean().sort_values(ascending=False)
    most_rated_data = all_data.groupby('title')['rating'].count().sort_values(ascending=False)
    all_data = pd.merge(all_data, highest_rated_data, on='title', how='left')
    all_data = pd.merge(all_data, most_rated_data, on='title', how='left')
    all_data = all_data.rename(columns={"rating":"num_ratings","rating_y":"avg_rating","rating_x":"rating"})

@app.route("/")
def login():
    return render_template("index.html")

@app.route("/receiver",methods=['POST'])
def receiver():
    global users
    global user_id
    user_id = str(request.form.get('userid'))
    for i in range(len(users)):
        if user_id ==str(users[i]):
            return redirect(url_for('profile'))
    else:
        return redirect(url_for('failed'))

@app.route("/createnewuser",methods=['POST'])
def create_new_user():
    return redirect(url_for('new_user'))

@app.route("/failed")
def failed():
    return render_template("index.html",message="Please enter a correct user id")

@app.route("/newuser")
def new_user():
    return render_template("newuser.html")

@app.route("/profile")
def profile():
    global best_books
    global user_id
    global num
    full_ratings = all_data[all_data['user_id'] == user_id]
    ratings = full_ratings[['title','rating']]
    ratings_titles = ratings[['title']]
    ratings = np.array(ratings)
    if ratings.size ==0:
        best_books = overall_best(num)
    else:
        print(num)
        best_books = matrix_factorization(full_ratings,num)
    return render_template("profile.html",ratings=ratings,best_books=best_books,num=num)

@app.route("/checkuser",methods=['POST'])
def check_user():
    global users
    global user_id
    user_id = str(request.form.get('userid'))
    for i in range(len(users)):
        if user_id ==str(users[i]):
            return redirect(url_for('existing_user'))
    else:
        users = np.append(users,[user_id])
        return redirect(url_for('profile'))

@app.route("/existinguser")
def existing_user():
    return render_template("newuser.html",message="Username already exists")

@app.route("/backhome",methods=['POST'])
def back_home():
    return redirect(url_for('login'))

@app.route("/toedit",methods=['POST'])
def to_edit():
    return redirect(url_for('edit'))

@app.route("/edit")
def edit():
    ratings = all_data[all_data['user_id'] == user_id]
    ratings = ratings[['title','rating']]
    ratings = np.array(ratings)
    return render_template("edit.html",ratings=ratings)

@app.route("/todelete",methods=['POST'])
def to_delete():
    return redirect(url_for('delete'))

@app.route("/delete")
def delete():
    ratings = all_data[all_data['user_id'] == user_id]
    ratings = ratings[['title','rating']]
    ratings = np.array(ratings)
    return render_template("delete.html",ratings=ratings)

@app.route("/toadd",methods=['POST'])
def to_add():
    return redirect(url_for('add'))

@app.route("/add")
def add():
    global user_id
    titles = all_data.title.unique()
    titles = np.sort(titles)
    return render_template("add.html",titles=titles)

@app.route("/number",methods=['POST'])
def number():
    global num
    num = int(request.form.get('n'))
    if num>995:
        num=995
    return redirect(url_for('profile'))

@app.route("/backtoprofile",methods=['POST'])
def back_to_profile():
    return redirect(url_for('profile'))

@app.route("/ratings",methods=['POST'])
def edit_ratings():
    global user_id
    global ratings_data
    title = str(request.form.get('titles'))
    rating = str(request.form.get('rating'))
    title_data = book_data[book_data['title'] == title]
    book_id = title_data.book_id.unique()
    book_index = ratings_data[ratings_data['book_id'] == book_id[0]].index.values.astype(int)
    user_index = ratings_data[ratings_data['user_id'] == user_id].index.values.astype(int)
    for index in book_index:
        if index in user_index:
            rating_index = index
            break
    ratings_data.at[rating_index,'rating'] = rating
    ratings_data.to_csv("ratings.csv",index=False)
    reinitialize()
    return redirect(url_for('profile'))

@app.route("/addrating",methods=['POST'])
def add_rating():
    global user_id
    title = str(request.form.get('titles'))
    rating = str(request.form.get('rating'))
    title_data = book_data[book_data['title'] == title]
    book_id = title_data.book_id.unique()
    row = [[user_id,book_id[0],rating]]
    new_row = pd.DataFrame(row,columns = ['user_id','book_id','rating'])
    with open('ratings.csv', 'a') as f:
        new_row.to_csv(f, header=False,index=False)
    f.close()
    reinitialize()
    return redirect(url_for('profile'))

@app.route("/deleterating",methods=['POST'])
def delete_rating():
    global user_id
    global ratings_data
    title = str(request.form.get('titles'))
    title_data = book_data[book_data['title'] == title]
    book_id = title_data.book_id.unique()
    book_index = ratings_data[ratings_data['book_id'] == book_id[0]].index.values.astype(int)
    user_index = ratings_data[ratings_data['user_id'] == user_id].index.values.astype(int)
    for index in book_index:
        if index in user_index:
            rating_index = index
            break
    ratings_data = ratings_data.drop([rating_index],axis=0)
    ratings_data.to_csv("ratings.csv",index=False)
    reinitialize()
    return redirect(url_for('profile'))

if __name__ == "__main__":
    app.run()
    
