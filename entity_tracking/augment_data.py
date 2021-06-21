import json
import os
from entity_augmentation import semantic_parse_entity_sentence, augment_entities_with_cpnet, get_candidate_entities

import time

from tqdm import tqdm

PATH_TO_DIRECTORY = os.path.dirname(os.path.dirname(__file__))


def augment_file_basic(input_file: str, output_file: str):
    """

    @param input_file: dataset file to augment
    @param output_file: location to write the augmented file
    @return: ?
    """
    full_input_path = os.path.join(PATH_TO_DIRECTORY, input_file)
    full_output_path = os.path.join(PATH_TO_DIRECTORY, output_file)

    num_lines = sum(1 for line in open(full_input_path))

    # print(f"Loading from previous attempt: starting at example {num_lines_already}")

    i = 0
    with open(full_input_path, 'r') as f, open(full_output_path, 'w+') as o:
        for line in tqdm(f, total=num_lines):
            obj = json.loads(line)
            entities, meta_data = read_line(obj)
            read_line(obj)
            time.sleep(.5)
            obj["entities"] = entities
            obj["knowledge"] = meta_data

            lines.append(obj)
            json.dump(obj, o)
            o.write("\n")

    # with open(full_output_path, 'w') as f:
    #     json.dump(lines, f)




def read_line(line: json):
    """

    @param line: json line to augment
    @return: augmented json

    format: original json + {"entities": list of entities, "knowledge":{entity: cpnet_triples}}
    """
    question = line["question"]

    initial_entities = get_candidate_entities(question)
    initial_entities_parsed = semantic_parse_entity_sentence(question)

    augmented_from_cpnet = augment_entities_with_cpnet(initial_entities, question)

    return augmented_from_cpnet


# input = "data/augmented_for_openpi/dev.jsonl"
input = "data/augmented_for_openpi/dev_unformatted.jsonl"
output = "data/augmented_for_openpi/dev_formatted.jsonl"
augment_file_basic(input, output)

aug_input = "data/formatted_for_gpt2/test.jsonl"
aug_output = "data/augmented_for_openpi/test_unformatted.jsonl"
aug_formatted = "data/augmented_for_openpi/test_formatted.jsonl"

# with open(os.path.join(PATH_TO_DIRECTORY, aug_input)) as f, open(os.path.join(PATH_TO_DIRECTORY, aug_output0)) as g:
#     i = 0
#     for line in g:
#         try:
#             b = json.loads(f.readline())
#             a = json.loads(line)
#             i += 1
#         except Exception as e:
#             print(i)
#             raise
#
#         if a["id"] != b["id"]:
#             print(i)
#             f.readline()
#


# augment_file_basic(aug_input, aug_output)

lines = []
with open(os.path.join(PATH_TO_DIRECTORY, aug_output), 'r') as f:
    for line in f:
        obj = json.loads(line)
        lines.append(obj)

with open(os.path.join(PATH_TO_DIRECTORY, aug_formatted), 'w') as f:
    for obj in lines:
        entities = obj["entities"]
        knowledge = obj["knowledge"]

        entities = [entity[0] for entity in entities]
        entities = list(dict.fromkeys(entities))
        knowledge = [v["relation"] for k, vs in knowledge.items() for v in vs]

        obj["entities"] = entities
        obj["knowledge"] = knowledge

        json.dump(obj, f)
        f.write("\n")

# knowledge = []
# with open(os.path.join(PATH_TO_DIRECTORY, fin_output), 'r') as f:
#     for line in f:
#         obj = json.loads(line)
#         knowledge.append(obj["knowledge"])
#
# print(sum([len(k) > 10 for k in knowledge]))
# over_ten = [k for k in knowledge if len(k) > 10]
# print(sum(len(k) for k in over_ten))
# print(max(len(k) for k in over_ten))

