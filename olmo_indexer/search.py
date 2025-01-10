from wimbd.es import es_init
from wimbd.es import get_documents_containing_phrases

# Make sure you have the config file with the API key first

dolma_idx = "docs_v1.5_2023-11-02"

es = es_init()

flipflop_mentions = get_documents_containing_phrases(dolma_idx, "Flip-Flop", num_documents=1000)

for mention in flipflop_mentions:
    print(mention['_source']['url'])