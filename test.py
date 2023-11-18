from gensim import corpora
from gensim.models import LdaModel
from gensim.test.utils import common_texts

# Step 1: Prepare the corpus
dictionary = corpora.Dictionary(common_texts)
print(dictionary)
print(common_texts)
corpus = [dictionary.doc2bow(text) for text in common_texts]

# Step 2: Train the LDA model
num_topics = 5
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

# Step 3: Inspect the topics
topics = lda_model.print_topics(num_words=5)
for topic in topics:
    print(topic)

# Step 4: Analyze a new document
new_doc = "Your new document text here"
new_doc_bow = dictionary.doc2bow(new_doc.split())
topic_distribution = lda_model.get_document_topics(new_doc_bow)
print("Topic Distribution for the new document:", topic_distribution)
