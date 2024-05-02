import argparse
import logging
import os

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from transformers import (MODEL_FOR_QUESTION_ANSWERING_MAPPING,
                          AdamW,
                          AutoConfig,
                          AutoModelForQuestionAnswering,
                          AutoTokenizer, )
from transformers.data.metrics.squad_metrics import (compute_predictions_log_probs,
                                                     compute_predictions_logits,
                                                     squad_evaluate, )
from transformers.data.processors.squad import SquadResult

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

MODEL_TYPES = tuple(conf.model_type for conf in MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())


def load_cache(app_args, evaluate=False):
    # Load data features from cache or dataset file
    input_dir = app_args.data_dir if app_args.data_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, app_args.model_name_or_path.split("/"))).pop(),
            str(app_args.max_seq_length),
        ),
    )

    # Init features and dataset from cache if it exists
    assert os.path.exists(cached_features_file), "No cached features found at {}".format(cached_features_file)
    logger.info("Loading features from cached file %s", cached_features_file)
    features_and_dataset = torch.load(cached_features_file)

    return features_and_dataset["dataset"], features_and_dataset["examples"], features_and_dataset["features"]


def train(app_args, model, tokenizer):
    train_dataset, _, _ = load_cache(app_args, evaluate=False)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=app_args.per_gpu_train_batch_size)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": app_args.weight_decay, },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=app_args.learning_rate, eps=app_args.adam_epsilon)

    # Train!
    logger.info("***** Running training *****")

    for epoch in range(1):
        evaluate(app_args, model, tokenizer)
        model.train()
        for step, batch in tqdm(enumerate(train_dataloader)):
            batch = tuple(t.to(app_args.device) for t in batch)
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "start_positions": batch[3],
                      "end_positions": batch[4]}

            outputs = model(**inputs)
            loss = outputs[0]
            # print(batch[0].shape)

            loss.backward()

            # if train_dataloader._elastic.is_sync_step():
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

            # print(loss.item())

        evaluate(app_args, model, tokenizer)

        # 05/12/2024 16:20:04 - INFO - __main__ -     Evaluation done in total 92.617401 secs (0.008550 sec per example)
        # OrderedDict([('f1', 87.12470988414496), ('total', 10570), ('best_f1_thresh', 0.0)])
        # 05/12/2024 16:20:40 - INFO - __main__ -   Saving model checkpoint to /home/cchen/yfliu/ares/data/squad/output

        # for key, value in results.items():
        #     tb_writer.add_scalar("eval_{}".format(key), value, epoch)
        # report_valid_metrics(epoch, 0.0, f1=results["f1"])


def evaluate(app_args, model, tokenizer):
    val_set, examples, features = load_cache(app_args, evaluate=True)
    val_sampler = SequentialSampler(val_set)
    val_loader = DataLoader(val_set, sampler=val_sampler, batch_size=app_args.per_gpu_eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation *****")

    all_results = []
    model.eval()
    with torch.no_grad():
        for step, batch in tqdm(enumerate(val_loader), desc="Evaluating"):
            batch = tuple(t.to(app_args.device) for t in batch)
            outputs = model(**{"input_ids": batch[0],
                               "attention_mask": batch[1],
                               "return_dict": False})

            for i, feature_index in enumerate(batch[3]):  # feature_indices = batch[3]
                output = [output[i].detach().cpu().tolist() for output in outputs]
                start_logits, end_logits = output
                unique_id = int(features[feature_index.item()].unique_id)
                result = SquadResult(unique_id, start_logits, end_logits)
                all_results.append(result)

    # Compute predictions
    predictions = compute_predictions_logits(examples,
                                             features,
                                             all_results,
                                             app_args.n_best_size,
                                             app_args.max_answer_length,
                                             app_args.do_lower_case,
                                             None,
                                             None,
                                             None,
                                             app_args.verbose_logging,
                                             app_args.version_2_with_negative,
                                             app_args.null_score_diff_threshold,
                                             tokenizer)
    results = squad_evaluate(examples, predictions)  # Compute the F1 and exact scores.
    print(results["f1"])
    logger.info("  Evaluation done")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", default="bert", type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_TYPES), )
    parser.add_argument("--model_name_or_path",
                        default="/home/cchen/yfliu/ares/data/squad/model/bert-base-uncased",
                        type=str,
                        help="Path to pretrained model or model identifier from huggingface.co/models", )
    parser.add_argument("--output_dir", default="/home/cchen/yfliu/ares/data/squad/output", type=str,
                        help="The output directory where the model checkpoints and predictions will be written.", )
    parser.add_argument("--data_dir", default="/home/cchen/yfliu/ares/data/squad/data", type=str,
                        help="The input data dir. Should contain the .json files for the task."
                             "If no data dir or train/predict files are specified, will run with tensorflow_datasets.")

    parser.add_argument("--version_2_with_negative", action="store_true",
                        help="If true, the SQuAD examples contain some that do not have an answer.", )
    parser.add_argument("--null_score_diff_threshold", type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.", )
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.", )
    parser.add_argument("--do_lower_case", action="store_true", default=True,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=12, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=2.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.", )
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another.", )
    parser.add_argument("--verbose_logging", action="store_true",
                        help="If true, all of the warnings related to data processing will be printed. A number of warnings are expected for a normal SQuAD evaluation.", )
    app_args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    app_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    app_args.model_type = app_args.model_type.lower()
    config = AutoConfig.from_pretrained(app_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(app_args.model_name_or_path, do_lower_case=app_args.do_lower_case)
    model = AutoModelForQuestionAnswering.from_pretrained(app_args.model_name_or_path, config=config).to(
        app_args.device)
    # print(model)
    # print(tokenizer)

    # Training
    train(app_args, model, tokenizer)

    # Save the trained model and the tokenizer
    logger.info("Saving model checkpoint to %s", app_args.output_dir)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(app_args.output_dir)
    tokenizer.save_pretrained(app_args.output_dir)
    # Load a trained model and vocabulary that you have fine-tuned
    model = AutoModelForQuestionAnswering.from_pretrained(app_args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(app_args.output_dir, do_lower_case=app_args.do_lower_case)
    model.to(app_args.device)


if __name__ == "__main__":
    main()
