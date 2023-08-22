# IMDb_movie_review_sentiment_analysis_using_NLP

For this project we will use a dataset of movie reviews from the IMDb (Internet Movie Database) website collected by Stanford researcher Andrew Maas. This dataset contains the text of the reviews, together with a label that indicates whether a review is “positive” or “negative.” The IMDb website itself contains ratings from 1 to 10. To simplify the modeling, this annotation is summarized as a two-class classification dataset where reviews with a score of 6 or higher are labeled as positive, and the rest as negative.

The data is first loaded and tokenized using count vectorizer. The data is then cleaned to remove stopwords. The number of tokens is reduced by taking only tokens that appear in atleast five documents. The data is then rescaled using tf-idf and a logistric regression model is then trained on its training set and predictions made on the test set. The most and least important features learned by the model are then plotted.

To improve upon the model, bag of wordss representation is then used.


<img width="889" alt="Screen Shot 2023-08-22 at 1 41 50 PM" src="https://github.com/mayank8893/Data_Science_and_ML_Projects/assets/69361645/d0b63361-a498-42bf-82f1-03da896612d6">
<img width="588" alt="Screen Shot 2023-08-22 at 1 41 56 PM" src="https://github.com/mayank8893/Data_Science_and_ML_Projects/assets/69361645/e917f941-18a5-405a-acaf-dd78ba79396b">
