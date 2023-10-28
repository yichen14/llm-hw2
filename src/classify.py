import argparse
import os
import tiktoken
import torch
from datasets import load_dataset
from model import DecoderLM
from omegaconf import OmegaConf
from tqdm import trange
from utils import determine_device, enable_tf32

YELP_TEMPLATE = "Here is a yelp review.\n{text}\nThis review is"
YELP_LABEL_MAP = {0: " negative", 1: " positive"}


@torch.inference_mode()
def score(
    model: DecoderLM,
    device: str,
    tokenizer: tiktoken.Encoding,
    texts: list[str],
    batch_size: int,
) -> torch.FloatTensor:
    """Scores all possible next tokens for the given texts

    Args:
        model: the language model
        device: device to put the tensors on
        tokenizer: the tokenizer
        texts: a list of strings for scoring
        batch_size: number of instances to score during one forward pass

    Returns:
        Logits corresponding to next token probabilities (B x V).

    
    Note: you should implement a batched version of this function by
        left-padding tokenized instances with `tokenizer.eot_token` so that all
        sequences have equal length. `attention_mask` should be set to 0.0 for
        padding tokens, and 1.0 everywhere else.
    """

    return ...


def classify_binary_sentiment(
    logits: torch.FloatTensor,
    tokens_of_interest: list[int],
    calibrate: bool = False,
) -> list[int]:
    """
    Args:
        logits: torch tensor corresponding to next token probabilities (B x V)
        tokens_of_interest: the indices for the tokens corresponding to negative
          and positive labels
        calibrate: when calibration is true, set the threshold according to your
          proposed calibration strategy in Question 3.6
    Returns:
        A list of predictions with length B, an element should be 0 if the
          negative class is more likely and 1 if the positive class is more
          likely.
    """

    probs = logits[:, tokens_of_interest].softmax(1)

    if calibrate:
        threshold = ...
    else:
        threshold = 0.5
    
    predictions = ...
    return predictions


def main():
    enable_tf32()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=OmegaConf.load,
        required=True,
        help="the yaml config file used for model training",
    )
    parser.add_argument("--calibrate", action="store_true")

    args = parser.parse_args()
    config = args.config

    # initialize tokenizer and model
    model_path = os.path.join(config.output_dir, "model.pt")
    assert os.path.exists(model_path), f"no model checkpoint at {model_path}"
    tokenizer = tiktoken.get_encoding(config.tokenizer_encoding)
    device = determine_device() if config.device == "auto" else config.device
    model = DecoderLM(tokenizer.n_vocab, **config.model_config).to(device)
    model.load_state_dict(torch.load(model_path))

    dataset = load_dataset("yelp_polarity")
    test_subset = (
        dataset["test"]
        .filter(
            lambda instance: len(
                tokenizer.encode(YELP_TEMPLATE.format(text=instance["text"]))
            )
            <= model.n_positions
        )
        .shuffle(seed=42)[:1000]
    )
    texts = [YELP_TEMPLATE.format(text=text) for text in test_subset["text"]]
    negative_token_id = tokenizer.encode_single_token(YELP_LABEL_MAP[0])
    positive_token_id = tokenizer.encode_single_token(YELP_LABEL_MAP[1])

    model.eval()
    logits = score(
        model,
        device,
        tokenizer,
        texts,
        config.batch_size,
    )

    predictions = classify_binary_sentiment(
        logits, [negative_token_id, positive_token_id], calibrate=args.calibrate
    )

    acc = sum(
        1 if pred == label else 0
        for pred, label in zip(predictions, test_subset["label"])
    ) / len(predictions)
    print(f"accuracy on yelp: {acc * 100:.1f}")


if __name__ == "__main__":
    main()
