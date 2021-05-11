import json

import textacy
import spacy
import textacy.ke

from typing import List, Dict, Any, Tuple

from conceptnet_util import get_closest_entities, embed_concept_sentence

tnlp = textacy.load_spacy_lang("en_core_web_sm")


def semantic_parse_entity_sentence(sent: str) -> List[str]:
    """

    @param sent: sentence to grab entities from
    @return: noun chunks that we consider "entities" to work with
    """
    doc = tnlp(sent)
    ents_ke = textacy.ke.textrank(doc, normalize="lemma")

    return ents_ke


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

def augment_entities_with_model():
    # TODO: Make more sophisticated entity prediction
    pass
