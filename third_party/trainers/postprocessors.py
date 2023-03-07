import abc
import numpy as np

class PostProcessor(abc.ABC):
    """Postprocess the predictions and labels to make them suitable for
    evaluation."""

    def __init__(self, tokenizer, ignore_pad_token_for_loss):
        self.tokenizer = tokenizer
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss

    def process(self, preds, labels, data_info=None):
        if isinstance(preds, tuple):
            preds = preds[0]
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels,
                            self.tokenizer.pad_token_id)
            preds = np.where(preds != -100, preds,
                            self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(
            preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        return decoded_preds, decoded_labels