The public datasets are extended with the toxic, moral, act, and VAD features 
 
toxic annotations consist of toxic and non_toxic

moral multi-labels are in the form of 0s and 1s list corresponding to ["care","harm","fairness","cheating","loyalty","betrayal","authority","subversion","purity","degradation"]

speech act labels belong to ['expression','others','question','statement','suggestion']
 
VAD features are the real-values obtained from eMFDScore Python library

Since, we share our dataset only with the tweet ids and annotations for privacy issues, it is possible to use fetchTweetFromID.py python script to fetch the tweet objects with the tweet id.
