
import re
import time
import threading
from typing import Dict, List, Tuple

import requests

from config.settings import settings
from utils.logger import get_logger


# Known accidental word joins observed in JARVIS responses.
KNOWN_JOINED_WORDS = (
    "thatlets",
    "youtoday",
    "justa",
    "matterthe",
    "yourapp",
    "aboutwhat",
    "formaximum",
    "consistencyacross",
    "andmaximize",
    "specificallythose",
    "onhigher",
    "returnsthis",
    "torun",
    "froma",
    "runsmoothly",
    "runsefficiently",
    "runsconsistently",
    "likea",
    "likean",
    "likethe",
    "anapp",
    "anapplication",
    "thesame",
    "thehost",
    "theapp",
    "thesystem",
    "theapplication",
    "machineissues",
)

# Valid technical terms that should not be flagged by themselves.
TECHNICAL_TERMS = {
    "dockerfile",
    "kubernetes",
    "localhost",
    "javascript",
    "typescript",
    "python",
    "fastapi",
    "ollama",
    "openai",
    "github",
    "gitlab",
    "devops",
    "websocket",
    "websockets",
    "http",
    "https",
    "api",
    "apis",
    "jwt",
    "sql",
    "mysql",
    "postgresql",
    "mongodb",
    "redis",
    "linux",
    "windows",
    "powershell",
    "uvicorn",
    "qwen3",
    "minilm",
}


def repair_response_spacing(text: str) -> str:
    """
    Repair clear punctuation-spacing problems.

    Does not split arbitrary words because doing so could damage
    technical terms, filenames, identifiers, and URLs.
    """
    if not isinstance(text, str):
        return ""

    text = text.strip()

    # Add a space after sentence-ending punctuation before a capital.
    text = re.sub(r"([.!?])([A-Z])", r"\1 \2", text)

    # Add a missing space after commas, semicolons, and colons.
    text = re.sub(r",(?=\S)", ", ", text)
    text = re.sub(r";(?=\S)", "; ", text)
    text = re.sub(r":(?=\S)", ": ", text)

    # Normalize repeated spaces and tabs without flattening newlines.
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def find_spacing_issues(text: str) -> List[str]:
    """
    Detect known suspicious joined-word patterns.

    This is a targeted detector, not a general-purpose spellchecker.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    text_lower = text.lower()
    issues: List[str] = []

    # Detect exact known joined tokens.
    for word in KNOWN_JOINED_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", text_lower):
            if word not in issues:
                issues.append(word)

    # Detect recurring joined-word patterns.
    patterns = (
        r"\bthat(?:lets|allows|makes|means|keeps)\b",
        r"\byou(?:today|can|should|will|need)\b",
        r"\byour(?:app|application|device|system)\b",
        r"\bno(?:matterthe|matterwhat)\b",
        r"\bconsistency(?:across|between)\b",
        r"\band(?:maximize|improve|increase)\b",
        r"\bspecifically(?:those|these)\b",
        r"\b(?:for|on|in|with|about|to|from)"
        r"(?:maximum|higher|those|these|run|a)\b",
        r"\brun(?:smoothly|efficiently|consistently)\b",
        r"\blike(?:a|an|the)\b",
        r"\ban(?:app|application|example)\b",
        r"\bthe(?:same|host|app|system|application)\b",
        r"\breturnsthis\b",
        r"\bmachine(?=[\"'])",
        
        r"\brun(?:the|a|an|this|your|it)\b",
        r"\bthat(?:lets|allows|makes|means|keeps)\b",

    )

    for pattern in patterns:
        for match in re.finditer(pattern, text_lower):
            token = match.group(0)

            if token not in issues:
                issues.append(token)

    # Avoid flagging known valid technical terms.
    return [
        issue
        for issue in issues
        if issue.lower() not in TECHNICAL_TERMS
    ]


def response_quality_score(text: str) -> int:
    """Return the number of detected spacing issues."""
    return len(find_spacing_issues(text))


class LLM:
    """
    Local Ollama adapter for JARVIS.

    Features:
    - Local Ollama inference.
    - Persistent HTTP connection reuse.
    - Configurable model and temperature.
    - Response timing and token metrics.
    - Connection and response error handling.
    - Conservative punctuation-spacing repair.
    - One retry for detected joined-word errors.
    - Serialized requests within this LLM instance.
    """

    def __init__(self):
        self.logger = get_logger("JARVIS.LLM")

        self.model = settings.AI_MODEL
        self.base_url = "http://127.0.0.1:11434"

        self.session = requests.Session()
        self.keep_alive = "10m"
        self.num_predict = 256
        self.timeout = (5, 180)

        self._request_lock = threading.Lock()

        self.logger.info(
            "Local LLM initialized | "
            f"model={self.model} | "
            f"keep_alive={self.keep_alive} | "
            f"num_predict={self.num_predict}"
        )

    def _request_ollama(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        num_predict: int,
    ) -> Tuple[str, dict]:
        """Send a chat request to Ollama and return text and metadata."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }

        with self._request_lock:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )

        response.raise_for_status()
        data = response.json()

        message = data.get("message") or {}
        content = message.get("content", "")

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(
                "Ollama returned an empty response. "
                "Check the selected model and Ollama logs."
            )

        return repair_response_spacing(content), data

    def _generate_with_spacing_retry(
        self,
        instructions: str,
        messages: List[Dict[str, str]],
    ) -> Tuple[str, dict, int]:
        """
        Generate an answer and retry once when known spacing errors
        are detected. Keep the candidate with fewer detected issues.
        """
        base_messages = [
            {"role": "system", "content": instructions}
        ]
        base_messages.extend(messages)

        content, data = self._request_ollama(
            messages=base_messages,
            temperature=settings.AI_TEMPERATURE,
            num_predict=self.num_predict,
        )

        original_issues = find_spacing_issues(content)

        if not original_issues:
            return content, data, 1

        self.logger.warning(
            "Suspicious word spacing detected | "
            f"patterns={original_issues[:8]} | retrying once"
        )

        retry_messages = list(base_messages)
        retry_messages.extend(
            [
                {
                    "role": "assistant",
                    "content": content,
                },
                {
                    "role": "user",
                    "content": (
                        
"Proofread your immediately preceding answer. Return the same "
"answer with correct English word spacing and punctuation. "
"Ensure every ordinary English word is separated correctly. "
"Specifically check for joined words such as 'thatlets', "
"'runthe', and 'likea'. Preserve valid technical terms, "
"names, code, URLs, filenames, and identifiers. Do not "
"change the meaning or add information. Return only "
"the corrected answer."

                    ),
                },
            ]
        )

        try:
            retry_content, retry_data = self._request_ollama(
                messages=retry_messages,
                temperature=0.2,
                num_predict=self.num_predict,
            )

            retry_content = repair_response_spacing(retry_content)
            retry_issues = find_spacing_issues(retry_content)

            # Accept only a response with fewer detected spacing issues.
            if len(retry_issues) < len(original_issues):
                self.logger.info(
                    "Spacing retry improved response | "
                    f"issues_before={len(original_issues)} | "
                    f"issues_after={len(retry_issues)}"
                )
                return retry_content, retry_data, 2

            self.logger.warning(
                "Spacing retry did not improve the detector score; "
                "keeping the original response."
            )

        except (
            requests.exceptions.RequestException,
            RuntimeError,
            ValueError,
        ) as exc:
            self.logger.warning(
                f"Spacing retry failed; keeping original response: {exc}"
            )

        return content, data, 1

    def generate(
        self,
        instructions: str,
        messages: List[Dict[str, str]],
    ) -> str:
        """
        Generate a response using Ollama.

        Preserves the interface used by other JARVIS modules:
        generate(instructions, messages) -> str
        """
        start_time = time.perf_counter()

        try:
            content, data, attempts = self._generate_with_spacing_retry(
                instructions=instructions,
                messages=messages,
            )

            content = repair_response_spacing(content)
            elapsed = time.perf_counter() - start_time

            total_duration = data.get("total_duration")
            load_duration = data.get("load_duration")
            prompt_tokens = data.get("prompt_eval_count")
            response_tokens = data.get("eval_count")

            metrics = [
                f"elapsed={elapsed:.2f}s",
                f"attempts={attempts}",
            ]

            if isinstance(total_duration, (int, float)):
                metrics.append(
                    f"ollama_total={total_duration / 1e9:.2f}s"
                )

            if isinstance(load_duration, (int, float)):
                metrics.append(
                    f"model_load={load_duration / 1e9:.2f}s"
                )

            if isinstance(prompt_tokens, int):
                metrics.append(f"prompt_tokens={prompt_tokens}")

            if isinstance(response_tokens, int):
                metrics.append(f"response_tokens={response_tokens}")

            self.logger.info(
                "LLM response generated | " + " | ".join(metrics)
            )

            remaining_issues = find_spacing_issues(content)

            if remaining_issues:
                self.logger.warning(
                    "Possible joined words remain | "
                    f"patterns={remaining_issues[:8]}"
                )

            return content

        except requests.exceptions.ConnectionError as exc:
            self.logger.error(f"Ollama connection failed: {exc}")
            raise RuntimeError(
                "Cannot connect to Ollama at "
                "http://127.0.0.1:11434. Make sure Ollama is running."
            ) from exc

        except requests.exceptions.Timeout as exc:
            self.logger.error("Ollama request timed out.")
            raise RuntimeError(
                "Ollama took too long to respond. "
                "Check model performance and request length."
            ) from exc

        except requests.exceptions.HTTPError as exc:
            status = (
                exc.response.status_code
                if exc.response is not None
                else "unknown"
            )
            details = (
                exc.response.text[:500]
                if exc.response is not None
                else str(exc)
            )

            self.logger.error(
                f"Ollama HTTP error | status={status} | "
                f"details={details}"
            )

            if status == 404:
                raise RuntimeError(
                    f"Ollama could not find model '{self.model}'. "
                    "Check available models with: ollama list"
                ) from exc

            raise RuntimeError(
                f"Ollama request failed with HTTP {status}: {details}"
            ) from exc

        except requests.exceptions.RequestException as exc:
            self.logger.error(f"Ollama request failed: {exc}")
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        except ValueError as exc:
            self.logger.error(f"Invalid JSON returned by Ollama: {exc}")
            raise RuntimeError(
                "Ollama returned an invalid JSON response."
            ) from exc

    def close(self) -> None:
        """Release the persistent HTTP connection pool."""
        self.session.close()
        self.logger.info("LLM HTTP session closed.")
