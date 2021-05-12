import json
import os
from entity_augmentation import semantic_parse_entity_sentence, augment_entities_with_cpnet

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
    lines = []
    with open(full_input_path, 'r') as f:
        for line in tqdm(f):
            obj = json.loads(line)
            entities, meta_data = read_line(obj)
            obj["entities"] = entities
            obj["knowledge"] = meta_data

            lines.append(obj)

    with open(full_output_path, 'w') as f:
        json.dump(lines, f)




def read_line(line: json):
    """

    @param line: json line to augment
    @return: augmented json

    format: original json + {"entities": list of entities, "knowledge":{entity: cpnet_triples}}
    """
    question = line["question"]

    initial_entities = semantic_parse_entity_sentence(question)
    augmented_from_cpnet = augment_entities_with_cpnet(initial_entities, question)

    return augmented_from_cpnet


input = "data/augmented_for_openpi/dev.jsonl"
output = "data/augmented_for_openpi/dev_unformatted.jsonl"

# augment_file_basic(input, output)

lines = []
with open(os.path.join(PATH_TO_DIRECTORY, input), 'r') as f:
    lines = json.load(f)

with open(os.path.join(PATH_TO_DIRECTORY, output), 'w') as f:
    for obj in lines:
        json.dump(obj, f)
        f.write("\n")
