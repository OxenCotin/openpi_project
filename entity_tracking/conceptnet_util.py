import json
from typing import List, Tuple, Any, Dict

import requests
import torch

from transformers import BertTokenizer, BertModel
from transformers import logging

from torch import functional as F

logger = logging.get_logger()

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

model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
logger.info("Loaded BERT model")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
logger.info(f"Loaded BERT tokenizer")


def query_conceptnet(word: str, attributes: List[str] = None) -> json:
    return requests.get(CONCEPT_NET_QUERY_HEADER + word).json()


def get_model():
    # This sucks
    return model


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


def get_closest_entities(entities: List[str], context: str, max_entities: int = 20, threshhold: float = 0) -> Tuple[
    List[Tuple[Any, Any]], Dict[str, list]]:
    """
    Iterates through a list of entities in context, looks up one-hop relations in conceptnet, and gets the entities
    at the other end of the relations that are 'most relevant'

    @param context: context entities appear in
    @param threshhold: threshhold of cpnet weight of entities to consider
    @type max_entities: max number of entities to track
    @param entities: tuples of entities present in the text to look up in conceptnet along with the context in which they appear

    @return list of entities most relevant to the procedural text,
    """
    possible_ents = []
    entity_dict = {}

    cosine = torch.nn.CosineSimilarity(dim=0)

    # For each entity, look up 'strong' relations in conceptnet. Add all above a certain threshhold to consideration\
    # import pdb
    # pdb.set_trace()
    for entity in entities:
        triples = query_conceptnet(format_cpnet_query(entity))['edges']
        embedded_context = embed_concept_sentence(context).view(-1)

        entity_dict[entity] = []
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
            concept = start if not start == entities else end
            text = format_cpnet_text(triple.get('surfaceText')) if triple.get('surfaceText') else \
                start + " " + relation_to_text[triple['rel']['@id']] + " " + end

            weight = triple['weight']

            # Simple heuristic for scoring the entities: more 'relevant' if concept triple is similar to original
            # context
            embedding, score = score_concept(cosine, embedded_context, text)

            out = {"entity": concept, "relation": text, "score": score}

            possible_ents.append((concept, score))
            entity_dict[entity].append(out)

    return sorted(possible_ents, key=lambda x: x[1], reverse=True), entity_dict


def score_concept(similarity, embedded_context, text):
    """

    @param similarity:
    @param embedded_context:
    @param text:
    @return:
    """
    embedding = embed_concept_sentence(text, model)
    score = similarity(embedding, embedded_context)
    score = score.item()

    return embedding, score


def embed_concept_sentence(sentence: str, model=model, embedder: str = 'BERT'):
    """

    @param model:
    @param embedder: model to use to embed triple text, defaults to pretrained bert
    @param sentence: concept_net triple to embed
    @return:
    """
    if embedder == "BERT":
        text, tokens, segments = bert_text_preparation(sentence, tokenizer)
        with torch.no_grad():
            outputs = model(tokens, segments)

            # first hidden state is input state
            hidden_states = outputs[2][1:]

        token_embeddings = hidden_states[-1]
        token_embeddings = torch.squeeze(token_embeddings, dim=0)
        list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]

        return torch.mean(token_embeddings, dim=0)
    else:
        raise NotImplementedError


def bert_text_preparation(text: str, tokenizer: BertTokenizer):
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(indexed_tokens)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensor = torch.tensor([segments_ids])

    return tokenized_text, tokens_tensor, segments_tensor


def json_to_text(trip: json):
    pass


sentence = "Knife is a tool"
embedded = embed_concept_sentence(sentence, model)

# print(embedded)
