import time
import queue
import logging
import threading

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from huggingface_hub.utils import logging as hf_logging

from config import MODEL, TOKEN, DEVICE, NUM_TURNS, MAX_NEW_TOKENS, MAX_INPUT_TOKENS
from dialog_manager import DialogManager
from gui import DialogWindow
from model_adapters import TransformersModelAdapter
from pipeline import VerbalTransformationPipeline
from test_cases import DEFAULT_TEST_CASE, get_test_case


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
hf_logging.set_verbosity_warning()


def log_step(message: str) -> float:
    logging.info(message)
    return time.time()


def load_model_and_tokenizer():
    if TOKEN:
        login(token=TOKEN, add_to_git_credential=False)
    else:
        logging.warning("No HF_TOKEN found.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, token=TOKEN, local_files_only=True)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, token=TOKEN)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    model_kwargs = {
        "token": TOKEN,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
    }

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            local_files_only=True,
            **model_kwargs,
        )
    except OSError:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            **model_kwargs,
        )

    model.to(DEVICE)
    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.use_cache = True

    return tokenizer, model


def conversation_worker(event_queue, tokenizer, model, test_case_key):
    test_case = get_test_case(test_case_key)
    model_adapter = TransformersModelAdapter(MODEL, tokenizer, model)
    pipeline = VerbalTransformationPipeline(model_adapter)
    manager = DialogManager(test_case, pipeline, NUM_TURNS)

    event_queue.put(("system", f"Model: {MODEL}"))
    event_queue.put(("system", f"Device: {DEVICE}"))
    event_queue.put(("system", f"Turns={NUM_TURNS}, max_new_tokens={MAX_NEW_TOKENS}, max_length={MAX_INPUT_TOKENS}"))
    manager.run(event_queue)


if __name__ == "__main__":
    selected_test_case = DEFAULT_TEST_CASE

    scenario = get_test_case(selected_test_case).scenario
    tokenizer, model = load_model_and_tokenizer()

    event_queue = queue.Queue()

    worker = threading.Thread(
        target=conversation_worker,
        args=(event_queue, tokenizer, model, selected_test_case),
        daemon=True,
    )
    worker.start()

    dialog = DialogWindow(event_queue, scenario)
    dialog.run()
