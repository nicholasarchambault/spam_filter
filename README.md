# Building a Naive Bayes Spam Filter
by Nicholas Archambault

Spam messaging through SMS is a problem that perpetuates a cyber arms race between spammers and programmers working to design increasingly more nuanced filters. This project aims to construct a simple SMS spam filter based on a naive interpretation of Bayesian probability.

## Goals
1. Clean dataset of SMS messages already classified as spam or non-spam.
2. Incorporate natural language processing techniques by creating dictionary of each individual word across the entire dataset and its usage frequency.
3. Use conditional Naive Bayes algorithm to calculate and update the probability a message is spam, given its content. Each word is assigned its own unique probability and, with Laplace smoothing, contributes to the overall classification of the message.
4. After training the filter on half the dataset, test it on the other half.

## Output
A spam filter built on simple natural language processing principles and the Naive Bayes algorithm which correctly classifies over 98% of new SMS messages.