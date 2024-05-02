import argparse
import os

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForQuestionAnswering, squad_convert_examples_to_features, \
    SquadV2Processor, SquadV1Processor

from models.bert.run_squad import MODEL_TYPES


def prepare_model(model_name="bert-base-uncased", model_dir="bert-base-uncased"):
    os.system(f"export HF_ENDPOINT=https://hf-mirror.com && "
              f"huggingface-cli download --resume-download {model_name} --local-dir {model_dir}")


def prepare_data(args, tokenizer, evaluate=False):
    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )

    print("Creating features from dataset file at %s", input_dir)

    processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
    if evaluate:
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_train_examples(args.data_dir)

    print(len(examples))
    print(examples[0])
    print(type(examples[0]))
    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=not evaluate,
        return_dataset="pt",
        threads=os.cpu_count(),
    )

    torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    return dataset, examples, features


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", default="bert", type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_TYPES), )
    parser.add_argument("--model_name_or_path", default="/home/cchen/yfliu/ares/data/squad/model/bert-base-uncased",
                        type=str,
                        help="Path to pretrained model or model identifier from huggingface.co/models", )
    parser.add_argument("--data_dir", default="/home/cchen/yfliu/ares/data/squad/data", type=str,
                        help="The input data dir. Should contain the .json files for the task."
                             "If no data dir or train/predict files are specified, will run with tensorflow_datasets.")
    parser.add_argument("--version_2_with_negative", action="store_true",
                        help="If true, the SQuAD examples contain some that do not have an answer.", )
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.", )
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.", )
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will be truncated to this length.", )
    parser.add_argument("--do_lower_case", action="store_true", default=True,
                        help="Set this flag if you are using an uncased model.")

    args = parser.parse_args()

    # prepare_model(model_dir=model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name_or_path, config=config)
    print(model)
    print(tokenizer)
    dataset, examples, features = prepare_data(args, tokenizer, evaluate=False)
    print(dataset)
    print(len(examples))
    print(len(features))
    dataset, examples, features = prepare_data(args, tokenizer, evaluate=True)
    print(dataset)
    print(len(examples))
    print(len(features))
