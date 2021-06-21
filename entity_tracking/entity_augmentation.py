import json
import os

import textacy
import spacy
import textacy.ke
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

from typing import List, Dict, Any, Tuple

from conceptnet_util import get_closest_entities, embed_concept_sentence

tnlp = textacy.load_spacy_lang("en_core_web_sm")
tnlp2 = textacy.load_spacy_lang("en_core_web_md")

similarity_fn = torch.nn.CosineSimilarity(dim=0)
SENTENCE_MODEL = SentenceTransformer("stsb-distilroberta-base-v2")


def get_candidate_entities(sent: str) -> List[str]:
    """

    @param context:
    @param sent: sentence to extract entities from
    @return:
    """
    parse = semantic_parse_entity_sentence(sent)
    augmented = augment_entities_with_cpnet(parse, sent)[0]
    augmented = list(set(a[0] for a in augmented))

    augmented = list(set(parse + augmented))
    other_filter = augmented.copy()
    sentence_filter = augmented.copy()

    # similarity = create_similarity_matrix(augmented)
    # similarity_model = create_similarity_matrix_model(other_filter)
    similarity_sentence = create_similarity_matrix_sentence_transformers(augmented)

    threshhold = .75

    # final = []
    #
    # for i in range(len(similarity)):
    #     for j in range(i):
    #         if i != j and similarity[i][j] > threshhold:
    #             augmented[j] = augmented[i]
    #
    # for i in range(len(similarity)):
    #     for j in range(i):
    #         if i != j and similarity_model[i][j] > threshhold:
    #             other_filter[j] = other_filter[i]

    for i in range(len(similarity_sentence)):
        for j in range(i):
            if i != j and similarity_sentence[i][j] > threshhold:
                sentence_filter[j] = sentence_filter[i]

    return list(set(sentence_filter))



def semantic_parse_entity_sentence(sent: str) -> List[str]:
    """

    @param sent: sentence to grab entities from
    @return: noun chunks that we consider "entities" to work with
    """
    doc = tnlp(sent)
    ents_ke = textacy.ke.textrank(doc, normalize="lemma")
    entities = [ent for ent, _ in ents_ke]

    return entities


def augment_entities_with_cpnet(entities: List[str], context: str) -> Tuple[List[Tuple[Any, Any]], Dict[str, list]]:
    """

    @param entities: starting entities under consideration
    @param context: context in which we are considering the entities
    @return: TODO: create datastructure? entity -> more entity + relations
    """
    return get_closest_entities(entities, context)


def combine_entities(entities: List[str], context: str) -> Dict[str, List[str]]:
    """

    @param entities: original candidate entities
    @param context: context in which the entities occur
    @return: entities with very likely repeats replaced
    """
    pass


def create_similarity_matrix(entities: List[str], context: str = ""):
    """

    @param entities: entities that we are comparing to each other
    @param context: context in which the entities appear
    @return: a similarity matrix of the entities (in context?)
    """
    matrix = np.zeros((len(entities), len(entities)))
    for i, ent in enumerate(entities):
        for j in range(i+1):
            matrix[i, j] = calculate_similarity_spacy(ent, entities[j])
    return matrix


def create_similarity_matrix_model(entities: List[str], context: str = ""):
    """

    @param entities: entities that we are comparing to each other
    @param context: context in which the entities appear
    @return: a similarity matrix of the entities (in context?)
    """

    matrix = np.zeros((len(entities), len(entities)))
    for i, ent in enumerate(entities):
        for j in range(i+1):
            matrix[i, j] = calculate_similarity_bert(ent, entities[j])
    return matrix


def create_similarity_matrix_sentence_transformers(entities: List[str], context: str = ""):
    """

    @param entities: entities that we are comparing to each other
    @param context: context in which the entities appear
    @return: a similarity matrix of the entities (in context?)
    """

    matrix = np.zeros((len(entities), len(entities)))
    for i, ent in enumerate(entities):
        for j in range(i+1):
            matrix[i, j] = calculate_similarity_sentence_transformers(ent, entities[j])
    return matrix

def calculate_similarity_bert(entity_1: str, entity_2: str) -> float:
    """

    @param entity_1:
    @param entity_2:
    @return: similarity between two entities (in the sense of whether they refer to the same thing
    TODO make this better than just embeddings?
    """
    embedded_1 = embed_concept_sentence(entity_1)
    embedded_2 = embed_concept_sentence(entity_2)

    return similarity_fn(embedded_1, embedded_2)


def calculate_similarity_spacy(entity1: str, entity2: str):
    tokens1 = tnlp(entity1)
    tokens2 = tnlp(entity2)

    # print(tokens1.similarity(tokens2))
    return tokens1.similarity(tokens2)


def calculate_similarity_sentence_transformers(entity1: str, entity2: str):
    embedding1 = torch.tensor(SENTENCE_MODEL.encode(entity1))
    embedding2 = torch.tensor(SENTENCE_MODEL.encode(entity2))

    return similarity_fn(embedding1, embedding2)

def augment_entities_with_model():
    # TODO: Make more sophisticated entity prediction
    pass


if __name__ == "__main__":
    # test = ["a car", "a town", "the stereos", "the car", "car", "stereo", "a state", "A tire", "A car", "An engine ", "A volvo", "Michigan", "Georgia", "California", "Maine", "New York", "Oregon", "texas", "Utah", "Colorado", "South Dakota", "New Mexico", "Massachusetts", "Arkansas", "Vermont", "Rhode Island"]

    # PATH_TO_DIRECTORY = os.path.dirname(os.path.dirname(__file__))
    # # print(PATH_TO_DIRECTORY)
    # META_FILE_PATH = os.path.join(PATH_TO_DIRECTORY, 'data/gold/dev/id_answers_metadata.jsonl')
    # GPT_FILE_PATH = os.path.join(PATH_TO_DIRECTORY, 'data/formatted_for_gpt2/dev.jsonl')
    #
    # store = []
    #
    # with open(GPT_FILE_PATH) as input_file:
    #     i = 0
    #     for line in input_file:
    #         obj = json.loads(line)
    #         question = obj["question"]
    #         parsed = semantic_parse_entity_sentence(question)
    #
    #         filtered_spacy, filter_bert = get_candidate_entities(question)
    #         store.append({"question": question, "parsed": parsed, "filtered_spacy": filtered_spacy, "filtered_bert": filter_bert})
    #         i += 1
    #         if i > 15:
    #             break
    #
    # with open(META_FILE_PATH) as input_file:
    #     i = 0
    #     for line in input_file:
    #         obj = json.loads(line)
    #         answers = obj["answers_metadata"]
    #         entities = [answer["entity"] for answer in answers]
    #
    #         store[i]["gold"] = entities
    #         i += 1
    #         if i > 15:
    #             break
    #
    # out_file = "tmp/eval/test_entities.csv"
    #
    # with open(os.path.join(PATH_TO_DIRECTORY, out_file), 'w') as f:
    #     json.dump(store, f, indent=2)
    #
    #
    get_candidate_entities(example1)