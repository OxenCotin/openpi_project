import json
import os
import fileinput
import spacy
import textacy
import textacy.ke
import requests

from typing import Sequence, List
import en_core_web_sm
from conceptnet_util import get_closest_entities, embed_concept_sentence

PATH_TO_DIRECTORY = os.path.dirname(os.path.dirname(__file__))
print(PATH_TO_DIRECTORY)
GPT_FILE_PATH = os.path.join(PATH_TO_DIRECTORY, 'data/formatted_for_gpt2/train.jsonl')
QUESTIONS_FILE_PATH = os.path.join(PATH_TO_DIRECTORY, 'data/gold/dev/id_question.jsonl')
META_FILE_PATH = os.path.join(PATH_TO_DIRECTORY, 'data/gold/dev/id_answers_metadata.jsonl')


def read_line(input_json, tokenizer, block_size, skip_answer, stop_token='<|endoftext|>'):
    metadata = {}
    tokenized_question = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_json['question']))
    if not skip_answer:
        if 'answer' not in input_json or input_json['answer'] == '':
            tokenized_answer = []
        else:
            tokenized_answer = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(input_json['answer'])) if not skip_answer else []
    else:
        tokenized_answer = []

    metadata['id'] = input_json['id']
    # OpenPIDataset._truncate_seq_pair(tokenized_question, tokenized_answer, max_length=block_size - 1)

    if not skip_answer:
        token_ids = tokenized_question + tokenized_answer + tokenizer.convert_tokens_to_ids([tokenizer.eos_token])
    else:
        token_ids = tokenized_question

    token_labels = [-100] * len(tokenized_question) + token_ids[len(tokenized_question):]

    import pdb
    pdb.set_trace()

    if len(token_ids) < block_size:
        add_tokens = block_size - len(token_ids)
        token_ids = token_ids + [0] * add_tokens
        token_labels = token_labels + [-100] * add_tokens

    if len(token_ids) > block_size:  # Truncate in block of block_size
        raise ValueError("Unexpected #tokens ({}) > block size ({}).".format(
            len(token_ids), block_size))

    assert len(token_ids) == len(token_labels)

    return token_ids, token_labels, metadata


questions = []
entities = []
with open(QUESTIONS_FILE_PATH) as f:
    for i, line in enumerate(f):
        input_json = json.loads(line)
        question = input_json['question']
        metadata = {}
        if i > 10:
            break
        questions.append(question)

with open(META_FILE_PATH) as f:
    for i, line in enumerate(f):
        if i > 10:
            break
        line_json = json.loads(line)
        answers = line_json["answers_metadata"]
        entity = [json["entity"] for json in answers]
        entities.append(set(entity))

examples = zip(questions, entities)
examples = list(examples)

nlp = en_core_web_sm.load()
# for example in examples:
#     doc = nlp(example)
#     # import pdb
#     # pdb.set_trace()
#     for ent in doc.ents:
#         print(ent.text, ent.start_char, ent.end_char, ent.label)

sent1 = "Squeeze a line of toothpaste onto one side of a soft sponge"
sent3 = "Vigorously rub the sponge in a circular motion over the entire surface of your headlight. Use a dry, " \
        "clean rag to wipe away any remaining toothpaste. Repeat every two to four months as needed. Now, " \
        "what happens? "
sent2 = "the USA supported the South Vietnamese in the Vietnam war"
sent4 = "Whisking gets rid of any streaks of yolk and whites in the final scramble"

tnlp = textacy.load_spacy_lang("en_core_web_sm")
#
# import pdb
# pdb.set_trace()
docs = [tnlp(doc) for doc, _ in examples]
# embedded_docs = [embed_concept_sentence(doc) for doc in docs]
entities = [tup[1] for tup in examples]

sov_test = [list(textacy.extract.subject_verb_object_triples(doc)) for doc in docs]
ent_test = [list(textacy.extract.noun_chunks(doc)) for doc in docs]
#
#
# import pdb
# pdb.set_trace()

ents_sg = [textacy.ke.sgrank(doc, normalize="lemma", include_pos="NOUN") for doc in docs]
ents_ke = [textacy.ke.textrank(doc, normalize="lemma", include_pos="NOUN") for doc in docs]
ents_ke_all = [textacy.ke.textrank(doc, normalize="lemma") for doc in docs]
ents_yake = [textacy.ke.yake(doc, normalize="lemma", include_pos="NOUN") for doc in docs]

ents_example = textacy.ke.textrank(tnlp(sent4), normalize="lemma", include_pos="NOUN")

terms = [doc._.to_bag_of_terms(entities=True, weighting="count", as_strings=True) for doc in docs]
terms1 = [doc._.to_terms_list(entities=True, weighting="count", as_strings=True) for doc in docs]


only_ents_ke = [[ent[0] for ent in ent_ke] for ent_ke in ents_ke_all]
ents_and_context = list(zip(only_ents_ke, questions))
all = [get_closest_entities(ent, context) for ent, context in zip(only_ents_ke, questions)]
inter = [[(ent[0], ent[2]) for ent in ents] for ents in all]

for i, ents in enumerate(inter):
    ents, context = ents
    ents.update(only_ents_ke[i])
    print(f"Prompt: {docs[i]}")
    print(f"Original entities: {only_ents_ke[i]}")
    print(f"Possible entities: {ents}")
    print(f"Correct entities: {entities[i]}")
    print(f"Missing: {entities[i].difference(ents)}")

# pdb.set_trace()

embedding_queries = [embed_concept_sentence(doc) for doc, _ in examples]

print(embedding_queries)

