from evaluate import load
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



def bertScoreMetric(prediction_list, reference_list):
    bertscore = load("bertscore")
    results = bertscore.compute(predictions=prediction_list, references=reference_list, lang="en")
    return results


def calculate_cosine_similarity(prediction_list, reference_list):
    """
    Calculate the cosine similarity between SBERT encodings of elements in prediction_list and reference_list.
    Args:
    prediction_list (list): List of predictions (strings).
    reference_list (list): List of references (strings).

    Returns:
    list: A list of cosine similarity scores.
    """
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    prediction_embeddings = model.encode(prediction_list)
    reference_embeddings = model.encode(reference_list)
    similarities = []
    for i in range(len(prediction_list)):
        similarity = cosine_similarity([prediction_embeddings[i]], [reference_embeddings[i]])[0][0]
        similarities.append(similarity)
    return similarities

"""predictions = ["Music albums", "Educational organizations"]
references = ["albums", "Research institutions"]
sbert_score = calculate_cosine_similarity(predictions,references )
bertScore = bertScoreMetric(predictions,references )
print(sbert_score)
print(bertScore)"""