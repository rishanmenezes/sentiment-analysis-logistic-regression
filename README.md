PROJECT TITLE
- Twitter Sentiment Analysis

PROJECT OVERVIEW
- This project builds a sentiment analysis model to classify tweets as either "Positive" or "Negative". Using a dataset of over 70,000 tweets associated with major brands and games (like Borderlands, Amazon, Google), the model analyzes the text to determine the user's emotional tone. This type of analysis is crucial for businesses to monitor brand reputation and customer feedback on social media.

DATASET DESCRIPTION
The project uses two datasets ("twitter_training.csv" and "twitter_validation.csv") with the following structure:
1. ID: A unique identifier for each tweet.
2. Topic: The subject of the tweet (e.g., specific companies or video games).
3. Sentiment: The label assigned to the tweet (Positive, Negative, Neutral, Irrelevant). *Note: For this specific analysis, only Positive and Negative tweets were used.*
4. Text: The actual content of the tweet.

OBJECTIVES OF THE PROJECT
- To preprocess raw text data by removing noise (stop words, punctuation) and normalizing it (stemming).
- To convert text data into numerical format using TF-IDF (Term Frequency-Inverse Document Frequency).
- To train a Logistic Regression model to classify the sentiment of new tweets.
- To evaluate the model's performance using accuracy scores and confusion matrices.
- To create a custom prediction function that can take any text input and return a sentiment label with an emoji.

STEPS PERFORMED
1. Data Loading: Imported the training and validation datasets using Pandas and renamed columns for clarity.
2. Data Preprocessing:
   - Handled missing values by removing rows with null text.
   - Filtered the dataset to focus only on "Positive" and "Negative" sentiments for binary classification.
   - Applied text cleaning techniques: lowercasing, tokenization, removing stop words, and stemming using NLTK.
3. Feature Engineering:
   - Used TF-IDF Vectorizer to transform the cleaned text into numerical feature vectors suitable for machine learning.
4. Model Training:
   - Split the data into training and testing sets (using the validation file as test data).
   - Trained a Logistic Regression model with a maximum iteration limit of 1000.
5. Evaluation & Visualization:
   - Calculated the accuracy score of the model.
   - Generated a Classification Report and a Confusion Matrix to analyze true positives/negatives.
   - Visualized the Confusion Matrix using a Seaborn heatmap.
6. Prediction System:
   - Developed a hybrid prediction function (`pred_text`) that combines rule-based keywords (for immediate negative detection) with the machine learning model predictions.

TOOLS AND LIBRARIES USED
- Python
- Pandas (for data manipulation)
- NumPy (for numerical operations)
- NLTK (Natural Language Toolkit for text processing)
- Scikit-learn (for model building and evaluation)
- Matplotlib & Seaborn (for data visualization)
- Jupyter Notebook

FILES INCLUDED
- Sentiment_Analysis.ipynb: The main notebook containing the code, model training, and testing logic.
- twitter_training.csv: The large dataset used to train the model.
- twitter_validation.csv: The smaller dataset used to validate the model's performance.

HOW TO RUN THE PROJECT
1. Install the required libraries:
   pip install pandas numpy nltk scikit-learn matplotlib seaborn
2. Download the Jupyter Notebook and the two CSV files into the same directory.
3. Open "Sentiment_Analysis.ipynb" in Jupyter Notebook or VS Code.
4. Run the cells sequentially. The notebook includes a step to download necessary NLTK data (stopwords, punkt) automatically.
5. Scroll to the bottom to use the `pred_text()` function and test your own sentences!

KEY INSIGHTS
- Text Preprocessing Impact: Cleaning the data (removing stopwords and stemming) significantly reduced the complexity of the dataset while retaining the core meaning.
- Model Performance: Logistic Regression proved to be an effective algorithm for this binary text classification task.
- Hybrid Approach: Combining a rule-based check (for specific "negative" keywords like "worst", "fail") with the ML model helped catch strong negative sentiments that the model might otherwise miss or misclassify.
- Visual Analysis: The confusion matrix heatmap provided a clear visual representation of where the model succeeded and where it made errors in classification.

CONCLUSION
- This project successfully demonstrates how to build a text classification pipeline from scratch. By combining standard NLP techniques with machine learning, we created a tool capable of automatically gauging public sentiment. This approach can be scaled to analyze real-time social media feeds for immediate brand insights.

AUTHOR
- Rishan Menezes
