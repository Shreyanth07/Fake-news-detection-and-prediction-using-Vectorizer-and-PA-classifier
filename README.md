# Fake-news-detection-and-prediction-using-Vectorizer-and-PA-classifier

- Data Exploration
- Data cleaning & Preparation
- Feature Extraction
- Model tesing & training

tf-idf term weighting as the feature to extract from these text.

Term frequency "tf" refers to the term frequency which indicates how often the terms can be found in documents. Tf's alone are often not sufficient as features as there are many commonly-used words such as "is", "are", "the", etc. which do not carry much information about the document hence, we do not want to take these terms as informative terms compared to others. These uninformative terms are actually referred to as stop words, and are often cleaned out during data cleaning/feature extraction as they do not hold much value in enhancing the model's ability to predict information.

Inverse document frequency "idf" is used to penalize common occurance terms across different contexts without adding any information. The equation for computing inverse document frequency is: idf(t)=log1+n1+df(t)+1. where n represents the total number of documents, t represents the term in question, df(t) represents the document frequency of that term (the number of documents within the set of documents containing that term for common terms such as "is", "are", etc.). idf(t) will most likely be 1, since all documents are highly likely to contain them (df(t)=n). The less a term occurs across different documents, the smaller the denominator will be, making the fraction bigger and in turn, idf(t) is also bigger.

tf-idf is the product of term-frequency and inverse document frequency, mathematically computed as: tf−idf(t,d)=tf(t,d)∗idf(t). Where d represents a document. The more commonly the word appears, the greater the value of tf will be, but if this is the case across different documents, it will be penalized with a small idf. On the other hand, a rarely-occurring word might have a smaller value of tf, but be highlighted by bigger idf values for not occurring often in different documents.

Initialize a TfidfVectorizer object that takes an input of a set of document strings and outputs of the normalized tf-idf vectors. By using fit_transform, we can fit the vectorizer to data and tranform them. Also, there is an option to use the max_df to indicate the cut-off document-frequency for stop words. We shall also set the cut-off document-frequency to be 0.7, which is the lowest possible value that the parameter can take. The final output of fitting & transforming data will give a sparse matrix with the size of n_samples by n_features (number of documents by number of unique words).
