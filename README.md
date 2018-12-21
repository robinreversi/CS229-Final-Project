# Whose Rap is it Anyways? Using Machine Learning to Determine Hip-Hop Artists from their Vocabulary

Stanford CS 229: Machine Learning -- Final Project (Autumn 2017-18)

Alex Wang, Robin Cheong, Vince Ranganathan

Updated 12/21/18

Poster available at: https://drive.google.com/file/d/1CWzrjjxoqvnZ1T9HSeU6vf7BQ96cFiZN/view?usp=sharing

Paper available at: http://cs229.stanford.edu/proj2017/final-reports/5198915.pdf

Here's the TL;DR:

- Author identification is a task of increasing relevance in machine learning.
- Algorithms have been developed to identify the authors in other contexts, e.g. literary texts, news articles, and tweets.
- Project aim: **identify the rapper of a song based only on lyrics, from a given set of rappers.**
- Parameters of the resulting model can be used to explore characteristics of rap lyrics, e.g. uniqueness of vocabulary or recurring lyrical themes.
- Best top 2 accuracy of 87.8% on the dev set

Our approaches to modelling the situation include:
- Multiclass Naive Bayes
- Softmax regression
- Neural networks
- One-vs-all Multiclass SVMs

For more about the sources of data, preprocessing techniques, classification models, performance levels, additional metrics (e.g. the importance matrix that identifies how indiciative a word _w_ is of rapper _r_), possible sources of error, and scope of future work, please look at the poster and paper available above.
