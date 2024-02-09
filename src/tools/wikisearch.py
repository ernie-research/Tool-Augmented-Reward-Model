import requests
'''
Wikipedia Search

Uses ColBERTv2 to retrieve Wikipedia documents.

input_query - A string, the input query (e.g. "what is a dog?")
k - The number of documents to retrieve

output - A list of strings, each string is a Wikipedia document

Adapted from Stanford's DSP: https://github.com/stanfordnlp/dsp/
Also see: https://github.com/lucabeetz/dsp
'''
class ColBERTv2:
    def __init__(self, url: str):
        self.url = url

    def __call__(self, query, k=10):
        topk = colbertv2_get_request(self.url, query, k)
        topk = [doc['text'] for doc in topk]
        return topk

def colbertv2_get_request(url: str, query: str, k: int):
    payload = {'query': query, 'k': k}
    res = requests.get(url, params=payload)

    topk = res.json()['topk'][:k]
    return topk

def wiki_search(
    input_query: str,
    url: str = 'http://index.contextual.ai:8893/api/search',
    k: int = 10
):
    retrieval_model = ColBERTv2(url)
    output = retrieval_model(input_query, k)
    return output