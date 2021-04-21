import json
import os

import spacy
import textacy
import requests

from typing import List, Sequence

CONCEPT_NET_QUERY_HEADER = "http://api.conceptnet.io/c/en/"

interested_relations = {"/r/PartOf", "/r/IsA", "/r/HasA", "/r/UsedFor", "/r/CapableOf", "/r/Desires",
                        "/r/AtLocation",
                        # "/r/HasSubevent", "/r/HasFirstSubevent", "/r/HasLastSubevent",
                        # "/r/HasPrerequisite",
                        # "/r/HasProperty", "/r/MotivatedByGoal", "/r/ObstructedBy", "/r/CreatedBy", "/r/Synonym",
                        # "/r/Causes", "/r/Antonym", "/r/DistinctFrom", "/r/DerivedFrom", "/r/SymbolOf", "/r/DefinedAs",
                        "/r/SimilarTo", "/r/Entails",
                        # "/r/MannerOf", "/r/RelatedTo",
                        # "/r/LocatedNear", "/r/HasContext", "/r/FormOf", "/r/EtymologicallyRelatedTo",
                        # "/r/EtymologicallyDerivedFrom", "/r/CausesDesire", "/r/MadeOf", "/r/ReceivesAction",
                        "/r/InstanceOf",
                        "/r/NotDesires", "/r/NotUsedFor", "/r/NotCapableOf", "/r/NotHasProperty", "/r/NotIsA",
                        "/r/NotHasA"}
# "/r/InheritsFrom", "/r/HasPainIntensity", "/r/DesireOf", "/r/LocationOfAction", "/r/NotMadeOf"}


relation_to_text = {
    "/r/PartOf": "is a part of",
    "/r/IsA": "is a",
    "/r/HasA": "has a",
    "/r/UsedFor": "is used for",
    "/r/CapableOf": "is capable of",
    "/r/Desires": "desires",
    "/r/AtLocation": "is at location",
    "/r/SimilarTo": "is similar to",
    "/r/Entails": "entails",
    "/r/InstanceOf": "is an instance of",
    "/r/NotUsedFor": "is not used for",
    "/r/NotCapableOf": "is not capable of",
    "/r/NotHasProperty": "does not have property",
    "/r/NotIsA": "is not a",
    "/r/NotHasA": "does not have"
}

""""
Select 'relevant' entities
"""


def query_conceptnet(word: str, attributes: List[str] = None) -> json:
    return requests.get(CONCEPT_NET_QUERY_HEADER + word).json()


def format_cpnet_text(text: str) -> str:
    """

    @param text: cpnet surfaceText format relationship text
    @return: formatted relation triple text
    """
    return text.replace('[', '').replace(']', '')


def format_cpnet_query(text: str) -> str:
    """

    @param text: entity to look up
    @return: remove whitespace, format
    """
    return text.replace(' ', '_')


def get_closest_entities(entities: List[str], max_entities: int = 20, threshhold: float = 0) -> List[str]:
    """
    Iterates through a list of entities in context, looks up one-hop relations in conceptnet, and gets the entities
    at the other end of the relations that are 'most relevant'

    @param threshhold: threshhold of cpnet weight of entities to consider
    @type max_entities: max number of entities to track
    @param entities: words present in the text to look up in conceptnet

    @return list of entities most relevant to the procedural text,
    """
    possible_ents = []

    # For each entity, look up 'strong' relations in conceptnet. Add all above a certain threshhold to consideration\
    # import pdb
    # pdb.set_trace()
    for entity in entities:
        triples = query_conceptnet(format_cpnet_query(entity))['edges']
        for triple in triples:
            # only look at relations we deem relevant and english triples
            if triple['rel']['@id'] not in interested_relations:
                continue
            elif triple['start']['language'] != "en" or triple['end']['language'] != "en":
                continue
            # Can set a threshhold for how strong the relationship is
            elif triple['weight'] < threshhold:
                continue
            start = triple['start']['label']
            end = triple['end']['label']
            concept = start if not start == entity else end
            text = format_cpnet_text(triple.get('surfaceText')) if triple.get('surfaceText') else \
                start + " " + relation_to_text[triple['rel']['@id']] + " " + end
            weight = triple['weight']

            possible_ents.append((concept, text, weight))

    # TODO: Embed triples and rank in context
    return sorted(possible_ents, key=lambda x: x[2])


def embed_concept_triple(triple: str, embedding):
    """

    @param triple: concept_net triple to embed
    @param embedding:
    @return:
    """
    pass


def json_to_text(trip: json):
    pass
