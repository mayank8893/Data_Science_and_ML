# IMDb_movie_review_sentiment_analysis_using_NLP

For this project we will use a dataset of movie reviews from the IMDb (Internet Movie Database) website collected by Stanford researcher Andrew Maas. This dataset contains the text of the reviews, together with a label that indicates whether a review is “positive” or “negative.” The IMDb website itself contains ratings from 1 to 10. To simplify the modeling, this annotation is summarized as a two-class classification dataset where reviews with a score of 6 or higher are labeled as positive, and the rest as negative.

The data is first loaded and tokenized using count vectorizer. The data is then cleaned to remove stopwords. The number of tokens is reduced by taking only tokens that appear in atleast five documents. The data is then rescaled using tf-idf and a logistric regression model is then trained on its training set and predictions made on the test set. The most and least important features learned by the model are then plotted.

To improve upon the model, bag of wordss representation is then used.
![Screenshot 2023-07-27 at 9 16 08 AM](https://github.com/mayank8893/IMDb_movie_review_sentiment_analysis_using_NLP/assets/69361645/bd87702f-c406-4423-9e34-66ee21ff3e19)

![Screenshot 2023-07-27 at 9 16 34 AM](https://github.com/mayank8893/IMDb_movie_review_sentiment_analysis_using_NLP/assets/69361645/90761419-c98f-4424-bb7f-c7fad433ab55)
