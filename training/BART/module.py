import transformers
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import pandas as pd
import numpy as np

import torch.nn.functional as F
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

import math
import random
import re
import argparse


class LitModel(pl.LightningModule):
    # Instantiate the model
    def __init__(self, learning_rate, tokenizer, model, hparams):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.learning_rate = learning_rate
        # self.freeze_encoder = freeze_encoder
        # self.freeze_embeds_ = freeze_embeds
        self.hparams = hparams

        if self.hparams.freeze_encoder:
            freeze_params(self.model.get_encoder())

        if self.hparams.freeze_embeds:
            self.freeze_embeds()

    def freeze_embeds(self):
        ''' freeze the positional embedding parameters of the model'''
        freeze_params(self.model.model.shared)
        for d in [self.model.model.encoder, self.model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)

    # Do a forward pass through the model
    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        # Load the data into variables
        src_ids, src_mask = batch[0], batch[1]
        tgt_ids = batch[2]
        # Shift the decoder tokens right (but NOT the tgt_ids)
        decoder_input_ids = shift_tokens_right(tgt_ids, tokenizer.pad_token_id)

        # Run the model and get the logits
        outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
        lm_logits = outputs[0]
        # Create the loss function
        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        # Calculate the loss on the un-shifted tokens
        loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):

        src_ids, src_mask = batch[0], batch[1]
        tgt_ids = batch[2]

        decoder_input_ids = shift_tokens_right(tgt_ids, tokenizer.pad_token_id)

        # Run the model and get the logits
        outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
        lm_logits = outputs[0]

        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        val_loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))

        return {'loss': val_loss}

    # Method that generates text using the BartForConditionalGeneration's generate() method
    def generate_text(self, text, eval_beams, early_stopping=True, max_len=40):
        ''' Function to generate text '''
        generated_ids = self.model.generate(
            text["input_ids"],
            attention_mask=text["attention_mask"],
            use_cache=True,
            decoder_start_token_id=self.tokenizer.pad_token_id,
            num_beams=eval_beams,
            max_length=max_len,
            early_stopping=early_stopping
        )
        return [self.tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in
                generated_ids]


def freeze_params(model):
    """ Function that takes a model as input (or part of a model) and freezes the layers for faster training
        adapted from finetune.py """
    for layer in model.parameters():
        layer.requires_grade = False


# Create a dataloading module as per the PyTorch Lightning Docs
class SummaryDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, data_file, batch_size, num_examples=20000):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_file = data_file
        self.batch_size = batch_size
        self.num_examples = num_examples

    # Loads and splits the data into training, validation and test sets with a 60/20/20 split
    def prepare_data(self):
        self.data = pd.read_csv(self.data_file)[:self.num_examples]
        self.train, self.validate, self.test = np.split(self.data.sample(frac=1),
                                                        [int(.6 * len(self.data)), int(.8 * len(self.data))])

    # encode the sentences using the tokenizer
    def setup(self, stage):
        self.train = encode_sentences(self.tokenizer, self.train['source'], self.train['target'])
        self.validate = encode_sentences(self.tokenizer, self.validate['source'], self.validate['target'])
        self.test = encode_sentences(self.tokenizer, self.test['source'], self.test['target'])

    # Load the training, validation and test sets in Pytorch Dataset objects
    def train_dataloader(self):
        dataset = TensorDataset(self.train['input_ids'], self.train['attention_mask'], self.train['labels'])
        train_data = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=self.batch_size)
        return train_data

    def val_dataloader(self):
        dataset = TensorDataset(self.validate['input_ids'], self.validate['attention_mask'], self.validate['labels'])
        val_data = DataLoader(dataset, batch_size=self.batch_size)
        return val_data

    def test_dataloader(self):
        dataset = TensorDataset(self.test['input_ids'], self.test['attention_mask'], self.test['labels'])
        test_data = DataLoader(dataset, batch_size=self.batch_size)
        return test_data


def shift_tokens_right(input_ids, pad_token_id):
    """ Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
        This is taken directly from modeling_bart.py
    """
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens


def encode_sentences(tokenizer, source_sentences, target_sentences, max_length=32, pad_to_max_length=True,
                     return_tensors="pt"):
    ''' Function that tokenizes a sentence
        Args: tokenizer - the BART tokenizer; source and target sentences are the source and target sentences
        Returns: Dictionary with keys: input_ids, attention_mask, target_ids
    '''

    input_ids = []
    attention_masks = []
    target_ids = []
    tokenized_sentences = {}

    for sentence in source_sentences:
        encoded_dict = tokenizer(
            sentence,
            max_length=max_length,
            padding="max_length" if pad_to_max_length else None,
            truncation=True,
            return_tensors=return_tensors,
            add_prefix_space=True
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    for sentence in target_sentences:
        encoded_dict = tokenizer(
            sentence,
            max_length=max_length,
            padding="max_length" if pad_to_max_length else None,
            truncation=True,
            return_tensors=return_tensors,
            add_prefix_space=True
        )
        # Shift the target ids to the right
        # shifted_target_ids = shift_tokens_right(encoded_dict['input_ids'], tokenizer.pad_token_id)
        target_ids.append(encoded_dict['input_ids'])

    target_ids = torch.cat(target_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": target_ids,
    }

    return batch


def noise_sentence(sentence_, percent_words, replacement_token="<mask>"):
    '''
    Function that noises a sentence by adding <mask> tokens
    Args: sentence - the sentence to noise
          percent_words - the percent of words to replace with <mask> tokens; the number is rounded up using math.ceil
    Returns a noised sentence
    '''
    # Create a list item and copy
    sentence_ = sentence_.split(' ')
    sentence = sentence_.copy()

    num_words = math.ceil(len(sentence) * percent_words)

    # Create an array of tokens to sample from; don't include the last word as an option because in the case of lyrics
    # that word is often a rhyming word and plays an important role in song construction
    sample_tokens = set(np.arange(0, np.maximum(1, len(sentence) - 1)))

    words_to_noise = random.sample(sample_tokens, num_words)

    # Swap out words, but not full stops
    for pos in words_to_noise:
        if sentence[pos] != '.':
            sentence[pos] = replacement_token

    # Remove redundant spaces
    sentence = re.sub(r' {2,5}', ' ', ' '.join(sentence))

    # Combine concurrent <mask> tokens into a single token; this just does two rounds of this; more could be done
    sentence = re.sub(r'<mask> <mask>', "<mask>", sentence)
    sentence = re.sub(r'<mask> <mask>', "<mask>", sentence)
    return sentence



checkpoint = ModelCheckpoint(filepath=base_dir + 'checkpoint_files_2/')
trainer = pl.Trainer(gpus = 1,
                     max_epochs = 1,
                     min_epochs = 1,
                     auto_lr_find = False,
                     checkpoint_callback = checkpoint,
                     progress_bar_refresh_rate = 500)