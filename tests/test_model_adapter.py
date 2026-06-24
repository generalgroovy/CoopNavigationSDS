from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import MagicMock, patch

from coop_navigation_sds.NaturalLanguageGeneration.models import (
    OllamaChatAdapter,
    OpenAICompatibleChatAdapter,
    TransformersModelAdapter,
    available_model_profile_keys,
    available_model_provider_keys,
    model_profile_metadata,
    model_adapter_runtime_metadata,
    model_provider_defaults,
    research_model_profiles_by_tier,
    ensure_ollama_ready,
)
from coop_navigation_sds.NaturalLanguageGeneration.model_runtime import create_model_adapter
from coop_navigation_sds.NaturalLanguageGeneration.model_runtime import MODEL_CACHE_DIR, _prepared_model


class FakeTensor:
    def __init__(self, values):
        self.values = values
        self.shape = (1, len(values))

    def to(self, _device):
        return self

    def __getitem__(self, item):
        if isinstance(item, tuple):
            return self
        return self.values[item]


class FakeGenerationConfig:
    def __init__(self):
        self.max_length = 2048
        self.max_new_tokens = None
        self.do_sample = False
        self.max_time = None
        self.temperature = None
        self.top_p = None


class FakeTokenizer:
    def __call__(self, _prompt, **_kwargs):
        tensor = FakeTensor([1, 2, 3])
        return {"input_ids": tensor, "attention_mask": tensor}

    def decode(self, _tokens, **_kwargs):
        return "reply"


class FakeModel:
    def __init__(self):
        self.generation_config = FakeGenerationConfig()
        self.call = None

    def generate(self, **kwargs):
        self.call = kwargs
        return [FakeTensor([1, 2, 3, 4])]


class TransformersModelAdapterTests(unittest.TestCase):
    def test_prepared_transformers_cache_is_independent_of_working_directory(self):
        resolved = Path(_prepared_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0"))

        self.assertTrue(Path(MODEL_CACHE_DIR).is_absolute())
        self.assertTrue(resolved.is_absolute())
        self.assertEqual(resolved.parent, Path(MODEL_CACHE_DIR))

    def test_ollama_preflight_accepts_installed_model(self):
        with patch(
            "coop_navigation_sds.NaturalLanguageGeneration.models._ollama_model_names",
            return_value=("llama3.2:latest",),
        ):
            status = ensure_ollama_ready(
                "http://127.0.0.1:11434/api",
                "llama3.2:latest",
            )
        self.assertEqual(status["model"], "llama3.2:latest")

    def test_ollama_preflight_reports_missing_model_before_dialogue(self):
        with patch(
            "coop_navigation_sds.NaturalLanguageGeneration.models._ollama_model_names",
            return_value=("another-model:latest",),
        ):
            with self.assertRaisesRegex(RuntimeError, "is not installed"):
                ensure_ollama_ready(
                    "http://127.0.0.1:11434/api",
                    "llama3.2:latest",
                )

    def test_generation_uses_only_generation_config_for_length_settings(self):
        model = FakeModel()
        adapter = TransformersModelAdapter(
            "fake",
            FakeTokenizer(),
            model,
            max_new_tokens=64,
        )
        fake_torch = SimpleNamespace(inference_mode=lambda: nullcontext())

        with patch.dict(sys.modules, {"torch": fake_torch}):
            reply = adapter.generate("prompt")

        self.assertEqual(reply, "reply")
        self.assertNotIn("max_new_tokens", model.call)
        self.assertIsNone(model.call["generation_config"].max_length)
        self.assertEqual(model.call["generation_config"].max_new_tokens, 64)

    def test_distinct_model_provider_families_and_conditions_are_public(self):
        self.assertEqual(
            available_model_provider_keys(),
            ("transformers", "openai_compatible", "ollama", "llama_cpp"),
        )
        self.assertTrue(model_provider_defaults("openai_compatible")["model_base_url"])
        self.assertTrue(model_provider_defaults("ollama")["model_base_url"])
        self.assertTrue(model_provider_defaults("llama_cpp")["model_base_url"])
        profiles = available_model_profile_keys()
        self.assertIn("tinyllama_1b_transformers", profiles)
        self.assertIn("qwen2_5_0_5b_llama_cpp", profiles)
        self.assertIn("gemma2_2b_ollama", profiles)
        self.assertIn("qwen3_4b_ollama", profiles)
        self.assertIn("mistral_7b_ollama", profiles)
        self.assertEqual(
            model_profile_metadata("qwen2_5_0_5b_llama_cpp")["provider"],
            "llama_cpp",
        )
        self.assertEqual(model_profile_metadata("gemma2_2b_ollama")["family"], "Gemma 2")
        self.assertEqual(model_profile_metadata("qwen3_4b_ollama")["provider"], "ollama")
        self.assertEqual(model_profile_metadata("mistral_7b_ollama")["size_tier"], "large")
        tiers = research_model_profiles_by_tier()
        self.assertEqual(set(tiers), {"small", "medium", "large"})
        self.assertTrue(all(len(profiles) == 2 for profiles in tiers.values()))
        self.assertEqual(
            tiers["medium"],
            ("llama3_2_3b_ollama", "phi3_3_8b_ollama"),
        )
        self.assertEqual(
            tiers["large"],
            ("qwen2_5_7b_ollama", "llama3_1_8b_ollama"),
        )
        self.assertEqual(
            model_profile_metadata(tiers["small"][0])["size_tier"],
            "small",
        )

    def test_factory_builds_local_llama_cpp_adapter_without_api_key(self):
        adapter = create_model_adapter(
            "llama_cpp",
            model_name="local-model",
            base_url="http://127.0.0.1:8080/v1",
        )

        self.assertIsInstance(adapter, OpenAICompatibleChatAdapter)
        self.assertEqual(adapter.name, "local-model")

    def test_unused_model_configuration_is_not_reported_as_an_executed_backend(self):
        metadata = model_adapter_runtime_metadata(None, provider="transformers")

        self.assertFalse(metadata["used"])
        self.assertEqual(metadata["provider"], "none")
        self.assertEqual(metadata["roles"], [])

    def test_factory_builds_openai_compatible_adapter(self):
        adapter = create_model_adapter(
            "openai_compatible",
            model_name="research-chat-model",
            api_key="test-key",
            base_url="http://example.test/v1",
        )

        self.assertIsInstance(adapter, OpenAICompatibleChatAdapter)
        self.assertEqual(adapter.name, "research-chat-model")

    def test_factory_builds_ollama_adapter(self):
        adapter = create_model_adapter(
            "ollama",
            model_name="research-local-model",
            base_url="http://127.0.0.1:11434/api",
        )

        self.assertIsInstance(adapter, OllamaChatAdapter)
        self.assertEqual(adapter.name, "research-local-model")

    def test_ollama_greedy_preset_disables_sampling(self):
        adapter = OllamaChatAdapter(model="research-local-model")
        response = MagicMock()
        response.read.return_value = b'{"message":{"content":"reply"}}'
        response.__enter__.return_value = response

        with patch(
            "coop_navigation_sds.NaturalLanguageGeneration.models.request.urlopen",
            return_value=response,
        ) as urlopen:
            adapter.generate("prompt")

        payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(payload["options"]["temperature"], 0.0)

    def test_factory_routes_transformers_configuration(self):
        sentinel = object()
        with patch(
            "coop_navigation_sds.NaturalLanguageGeneration.model_runtime.create_transformers_adapter",
            return_value=sentinel,
        ) as factory:
            adapter = create_model_adapter(
                "transformers",
                model_name="local-model",
                device="cpu",
                max_new_tokens=32,
                max_input_tokens=1024,
                allow_model_download=False,
            )

        self.assertIs(adapter, sentinel)
        factory.assert_called_once_with(
            model_name="local-model",
            device="cpu",
            max_new_tokens=32,
            max_input_tokens=1024,
            allow_model_download=False,
        )
