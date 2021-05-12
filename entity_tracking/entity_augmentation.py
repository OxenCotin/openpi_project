import json

import textacy
import spacy
import textacy.ke
import torch

from typing import List, Dict, Any, Tuple

from conceptnet_util import get_closest_entities, embed_concept_sentence

tnlp = textacy.load_spacy_lang("en_core_web_sm")
similarity_fn = torch.nn.CosineSimilarity(dim=0)

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


def create_similarity_matrix(entities: List[str], context: str) -> Array[str]:
    """

    @param entities: entities that we are comparing to each other
    @param context: context in which the entities appear
    @return: a similarity matrix of the entities (in context?)
    """
    embedded


def calculate_similarity(entity_1: str, entity_2: str) -> float:
    """

    @param entity_1:
    @param entity_2:
    @return: similarity between two entities (in the sense of whether they refer to the same thing
    TODO make this better than just embeddings?
    """
    embedded_1 = embed_concept_sentence(entity_1)
    embedded_2 = embed_concept_sentence(entity_2)

    return similarity_fn(embedded_1, embedded_2)


def augment_entities_with_model():
    # TODO: Make more sophisticated entity prediction
    pass
