from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import Trainer, TrainingArguments

# dataset = ...  # some Datasets object with train/validation split and columns 'text' and 'summary'
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')


def convert_to_features(example_batch):
    input_encodings = tokenizer.batch_encode_plus(example_batch['text'], pad_to_max_length=True, max_length=1024,
                                                  truncation=True))
    target_encodings = tokenizer.batch_encode_plus(example_batch['summary'], pad_to_max_length=True, max_length=1024,
                                                   truncation=True))

    labels = target_encodings['input_ids']
    decoder_input_ids = shift_tokens_right(labels, model.config.pad_token_id)
    labels[labels[:, :] == model.config.pad_token_id] = -100

    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'decoder_input_ids': decoder_input_ids,
        'labels': labels,
    }

    return encodings


dataset = dataset.map(convert_to_features, batched=True)
columns = ['input_ids', 'labels', 'decoder_input_ids', 'attention_mask', ]
dataset.set_format(type='torch', columns=columns)

training_args = TrainingArguments(
    output_dir='./models/bart-summarizer',
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation']
)