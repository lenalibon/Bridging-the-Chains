import torch
import re


def extract_last_qa_pair(tokenizer, chain_tokenized: torch.LongTensor, stop_string='.",'):
    decoded = tokenizer.decode(chain_tokenized[0], skip_special_tokens=False)

    qa_blocks = [block.strip() for block in decoded.split("Q:") if block.strip()]
    last_qa_raw = qa_blocks[-1]

    # Escape the stop string for safe regex use
    escaped_stop = re.escape(stop_string)

    # Replace the stop string ONLY if it appears at the very end (possibly with whitespace)
    last_qa_raw = re.sub(rf'{escaped_stop}\s*$', '."]', last_qa_raw)

    last_qa_raw = last_qa_raw.replace("\nA: \n", "\nA: ")
    last_qa_full = "Q: " + last_qa_raw
    return last_qa_full


def prompt_intermediate_answer(tokenizer, chain, stop_string='.",'):
    """
    Given a non-completed chain, create a prompt for the model to output an intermediate answer.
    """
    qa_prompt = extract_last_qa_pair(tokenizer, chain, stop_string)

    return ("Answer the following question directly. Do not continue the reasoning process.\n\n"
            + qa_prompt.strip()
            + "\nThat means the answer is "
            )
