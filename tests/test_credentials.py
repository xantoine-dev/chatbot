"""Execute the application with offline UI and API boundary doubles."""

from contextlib import nullcontext
import os
from pathlib import Path
import runpy
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch


APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


class CredentialTests(unittest.TestCase):
    def setUp(self):
        # Each test owns its synthetic environment; sessions within a test share it.
        environment = patch.dict(os.environ, {"TEST_MARKER": "unchanged"}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def run_session(self, key="", provider="Gemini (Google)", state=None):
        ui = MagicMock()
        ui.session_state = {} if state is None else state
        ui.radio.return_value = provider
        ui.text_area.return_value = " Be helpful "
        ui.text_input.return_value = key
        ui.chat_input.side_effect = lambda *args, **kwargs: (
            None if kwargs.get("disabled") else "Hello"
        )
        ui.chat_message.side_effect = lambda *args: nullcontext()
        ui.write_stream.side_effect = lambda stream: "".join(stream)

        gemini = MagicMock()
        gemini.return_value.models.generate_content.return_value = SimpleNamespace(
            text="Gemini reply"
        )
        openai = MagicMock()
        openai.return_value.chat.completions.create.return_value = iter(["OpenAI reply"])
        google_module = ModuleType("google")
        google_module.genai = SimpleNamespace(Client=gemini)
        openai_module = ModuleType("openai")
        openai_module.OpenAI = openai
        with patch.dict("sys.modules", {
            "streamlit": ui, "google": google_module, "openai": openai_module
        }):
            runpy.run_path(str(APP), run_name="__main__")
        return ui, gemini, openai

    def test_entered_key_is_explicit_and_environment_is_unchanged(self):
        # Catches implicit SDK credentials and process-global session writes.
        os.environ["GOOGLE_API_KEY"] = "synthetic-other-server-key"
        before = dict(os.environ)
        ui, gemini, openai = self.run_session("synthetic-session-key")
        self.assertEqual(dict(os.environ), before)
        gemini.assert_called_once_with(api_key="synthetic-session-key")
        gemini.return_value.models.generate_content.assert_called_once_with(
            model="gemini-2.5-flash",
            contents=[
                {"role": "user", "parts": [{"text": "System instructions: Be helpful"}]},
                {"role": "user", "parts": [{"text": "Hello"}]},
            ],
        )
        self.assertEqual(ui.session_state["Gemini (Google)_messages"][-1],
                         {"role": "assistant", "content": "Gemini reply"})
        openai.assert_not_called()

    def test_blank_second_session_cannot_inherit_first_sessions_key(self):
        self.run_session("synthetic-first-session")
        ui, gemini, _ = self.run_session()
        gemini.assert_not_called()
        ui.chat_input.assert_not_called()
        self.assertEqual(ui.session_state["Gemini (Google)_messages"], [])

    def test_configured_server_fallback_is_passed_explicitly(self):
        os.environ["GEMINI_API_KEY"] = "synthetic-server-key"
        before = dict(os.environ)
        _, gemini, _ = self.run_session()
        gemini.assert_called_once_with(api_key="synthetic-server-key")
        self.assertEqual(dict(os.environ), before)

    def test_entered_key_overrides_server_fallback_without_replacing_it(self):
        os.environ["GEMINI_API_KEY"] = "synthetic-server-key"
        before = dict(os.environ)
        _, entered, _ = self.run_session("synthetic-entered-key")
        _, fallback, _ = self.run_session()
        entered.assert_called_once_with(api_key="synthetic-entered-key")
        fallback.assert_called_once_with(api_key="synthetic-server-key")
        self.assertEqual(dict(os.environ), before)

    def test_missing_key_never_constructs_client_or_requests_content(self):
        ui, gemini, _ = self.run_session()
        gemini.assert_not_called()
        ui.chat_input.assert_not_called()
        ui.info.assert_called_once()

    def test_openai_keeps_explicit_key_streaming_and_separate_history(self):
        state = {"Gemini (Google)_messages": [{"role": "user", "content": "Private"}]}
        before = dict(os.environ)
        ui, gemini, openai = self.run_session("synthetic-openai-key", "OpenAI", state)
        openai.assert_called_once_with(api_key="synthetic-openai-key")
        openai.return_value.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hello"},
            ],
            stream=True,
        )
        self.assertEqual(ui.session_state["OpenAI_messages"][-1],
                         {"role": "assistant", "content": "OpenAI reply"})
        self.assertEqual(state["Gemini (Google)_messages"],
                         [{"role": "user", "content": "Private"}])
        gemini.assert_not_called()
        self.assertEqual(dict(os.environ), before)


if __name__ == "__main__":
    unittest.main()
