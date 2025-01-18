from transformers import LogitsProcessorList, LogitsProcessor
import torch

def preprocess_qa(example):
    options = [
        f"{label}: {text}," 
        for label, text in zip(example['choices']['label'], example['choices']['text'])
    ]
    options_text = " ".join(options)
    
    # Formatting the input and target text for fine-tuning
    return {
        "text": (
            f"Question: {example['question']}. "
            f"Options: {options_text}. "
            "Return only the letter corresponding to the correct answer (A, B, C, D, or E). The answer is "
        ),
        "target_text": (
            example['answerKey']
        )
    }


# Custom LogitsProcessor for valid tokens restriction
class RestrictToValidTokens(LogitsProcessor):
    def __init__(self, valid_tokens):
        self.valid_tokens = valid_tokens

    def __call__(self, input_ids, scores):
        # Mask all logits except the valid tokens
        mask = torch.full_like(scores, float('-inf'))
        mask[..., self.valid_tokens] = 0
        return scores + mask