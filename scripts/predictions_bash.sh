#!/bin/bash

# Loading the required module
module load anaconda/2021a

TRAINED_MODEL=${1-"/openpi_project/openpi_project/tmp/training_output/gpt2_augmented"}
INPUT_TO_PRED_CSV=${2-"/openpi_project/openpi_project/data/augmented_for_openpi/test_formatted.jsonl"}
OUTPUT_FILEPATH_CSV=${3-"/openpi_project/openpi_project/tmp/eval/gpt2_augmented_max10_samples/out.csv"}
AGG_OUTPUT_FILEPATH_CSV=${4-"/openpi_project/openpi_project/tmp/eval/gpt2_augmented_max10_samples/agg_long.csv"}
MAX_LEN=${5-120}

set -x  # print the command being executed.

IFS=',' read -ra prediction_input_files <<< "$INPUT_TO_PRED_CSV"
IFS=',' read -ra prediction_output_files <<< "$OUTPUT_FILEPATH_CSV"
IFS=',' read -ra agg_pred_output_files <<< "$AGG_OUTPUT_FILEPATH_CSV"

for i in ${!prediction_input_files[*]}; do
  python training/generation.py \
      --model_path "$TRAINED_MODEL" \
      --test_input_file "${prediction_input_files[i]}" \
      --unformatted_outpath "${prediction_output_files[i]}" \
      --formatted_outpath "${agg_pred_output_files[i]}" \
      --max_len $MAX_LEN
done
