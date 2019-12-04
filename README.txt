To run the system, navigate to the directory in the command line, run it with python server.py 
and open a browser window to http://localhost:5000/
Enter a number from 1-1000 to use an existing user profile or create your own with new user. 
The options to add, edit and delete from the profile are at the bottom of the page.
Requirements: Python 3 or above, pip install numpy, pandas, scipy, flask

Description:

A large dataset has been adapted from the https://github.com/zygmuntz/goodbooks-10k dataset. 
The adapted version contains 3213 of the most popular books, 1000 users and 60898 ratings.
Both tables are updated dynamically when the user adds, deletes or edits ratings.

The profile page of the system shows the previous ratings of the user (title and rating)
and is dynamically updated as the ratings are edited, added and deleted.
If the user has no previous ratings, the recommendations are the top rated books for all users.
The user can choose the number of recommendations to view and they are displayed as their titles 
with the average user rating, in order of the best predictive scores.

The recommendation algorithm used is matrix factorization, to support the large dataset,
using the top 50 features and incorporating the user's previous ratings to get the best predictions.
Only the books with the number of recommendations in the top third for all books are presented to the user
to ensure a sufficient number of ratings for a trustworthy prediction. 
This does not include any of the books the user has previously rated.
The web-based system uses Flask and communication with the server to support advanced user interactions.
