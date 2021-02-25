import os
import torch
import transformers
import jax
import jax.numpy as jnp

from datasets import load_dataset
from transformers import EvalPrediction
from transformers import RobertaTokenizerFast
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from callbacks.PerplexityCallback import PerplexityCallback
from callbacks.CarbonTrackerCallback import CarbonTrackerCallback


def main():
    epochs = 1
    dataset = load_dataset('cc_news', script_version='master')
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    dataset_reduced = dataset['train']['text'][:10]
    del dataset

    inputs = tokenizer.batch_encode_plus(
        dataset_reduced, truncation=True, padding=True, verbose=True, max_length=512
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )


    config = RobertaConfig.from_pretrained('./trained_model/config.json')
    model = RobertaForMaskedLM(config)
    model.load_state_dict(torch.load('./trained_model/pytorch_model.bin'))


    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=epochs,
        per_device_eval_batch_size=1,
        logging_dir='./logs',
        eval_accumulation_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=inputs['input_ids'],
        eval_dataset=inputs['input_ids'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[CarbonTrackerCallback(epochs), PerplexityCallback()],
    )

    with torch.no_grad():
        trainer.evaluate()


def compute_metrics(logits, labels, weights, label_smoothing=0.0):
    """Compute summary metrics."""
    loss, normalizer = cross_entropy(logits, labels, weights, label_smoothing)
    acc, _ = accuracy(logits, labels, weights)
    metrics = {"loss": loss, "accuracy": acc, "normalizer": normalizer}
    return metrics


def accuracy(logits, targets, weights=None):
    """Compute weighted accuracy for log probs and targets.
    Args:
     logits: [batch, length, num_classes] float array.
     targets: categorical targets [batch, length] int array.
     weights: None or array of shape [batch, length]
    Returns:
      Tuple of scalar loss and batch normalizing factor.
    """
    if logits.ndim != targets.ndim + 1:
        raise ValueError(
            "Incorrect shapes. Got shape %s logits and %s targets" % (str(logits.shape), str(targets.shape))
        )

    loss = jnp.equal(jnp.argmax(logits, axis=-1), targets)
    loss *= weights

    return loss.sum(), weights.sum()


def cross_entropy(logits, targets, weights=None, label_smoothing=0.0):
    """Compute cross entropy and entropy for log probs and targets.
    Args:
     logits: [batch, length, num_classes] float array.
     targets: categorical targets [batch, length] int array.
     weights: None or array of shape [batch, length]
     label_smoothing: label smoothing constant, used to determine the on and off values.
    Returns:
      Tuple of scalar loss and batch normalizing factor.
    """
    if logits.ndim != targets.ndim + 1:
        raise ValueError(
            "Incorrect shapes. Got shape %s logits and %s targets" % (str(logits.shape), str(targets.shape))
        )

    vocab_size = logits.shape[-1]
    confidence = 1.0 - label_smoothing
    low_confidence = (1.0 - confidence) / (vocab_size - 1)
    normalizing_constant = -(
        confidence * jnp.log(confidence) + (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20)
    )
    soft_targets = common_utils.onehot(targets, vocab_size, on_value=confidence, off_value=low_confidence)

    loss = -jnp.sum(soft_targets * log_softmax(logits), axis=-1)
    loss = loss - normalizing_constant

    if weights is not None:
        loss = loss * weights
        normalizing_factor = weights.sum()
    else:
        normalizing_factor = np.prod(targets.shape)

    return loss.sum(), normalizing_factor


if __name__ == '__main__':
    main()

