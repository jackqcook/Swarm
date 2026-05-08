"""
drone/command_interpreter.py
─────────────────────────────
Tier 1: Interprets vague natural-language commands using a small VLM
and decides whether to execute or ask for clarification.

Architecture:
  1. VLM (PaliGemma2-3B / SmolVLM-2B / Qwen2.5-VL-3B) generates
     a structured multiple-choice prediction set (options A–D + E=ask).
  2. KnowNo-style conformal prediction gating:
     - If prediction set is a singleton → execute.
     - If |set| > 1 → ask the operator a clarifying question.
  3. Output: InterpretResult with either a Goal or a clarification question.

On Jetson Orin Nano 8GB (FP4):
  - PaliGemma2-3B: ~21 tok/s → ~4s for an 80-token plan (acceptable for Tier 1)
  - SmolVLM-2B:    ~13 tok/s → ~6s

For development on CPU/GPU desktop, any HuggingFace VLM works the same way.
Swap the model_id in configs/sim_default.yaml to test different models.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class DroneGoal:
    """
    Structured output from the interpreter.
    The planner converts this into concrete waypoints.
    """
    action: str                    # e.g. "fly_to", "hover", "survey", "return"
    description: str               # Human-readable summary
    target_lat: Optional[float] = None
    target_lon: Optional[float] = None
    target_alt_m: Optional[float] = None
    region_description: Optional[str] = None  # "northeast clearing", "the barn"
    constraints: List[str] = field(default_factory=list)  # ["avoid trees", "stay under 50m"]
    raw_intent: Optional[str] = None


@dataclass
class InterpretResult:
    needs_clarification: bool
    goal: Optional[DroneGoal] = None
    clarification_question: Optional[str] = None
    confidence: float = 0.0
    options_considered: List[str] = field(default_factory=list)


# ── KnowNo-style conformal prediction ─────────────────────────────────────────

class KnowNoGate:
    """
    Lightweight conformal prediction gating.

    Calibration:
      - Collect (command, VLM output, ground-truth) on a small dev set.
      - Set tau so that the prediction set covers ground-truth >= target_coverage.

    At inference:
      - Include option i in the set if softmax_prob[i] >= tau.
      - If |set| == 1 → execute. If |set| > 1 or E in set → ask.

    Without a calibration dataset, we use a sensible default tau=0.6,
    which means we ask for help if no single option has ≥60% probability.
    """

    def __init__(self, tau: float = 0.6, target_coverage: float = 0.85):
        self.tau = tau
        self.target_coverage = target_coverage

    def decide(self, option_probs: dict[str, float]) -> tuple[bool, list[str]]:
        """
        Returns (needs_clarification, prediction_set).

        option_probs: {'A': 0.72, 'B': 0.18, 'C': 0.05, 'D': 0.03, 'E': 0.02}
        E always = "ask for help"
        """
        prediction_set = [opt for opt, p in option_probs.items() if p >= self.tau]

        # If E (ask-for-help) made it in, or multiple options, ask
        needs_clarification = len(prediction_set) != 1 or "E" in prediction_set

        return needs_clarification, prediction_set

    def calibrate(self, calibration_data: list) -> float:
        """
        Given a list of (probs_dict, correct_option) tuples,
        find the tau that achieves target_coverage.
        (Placeholder — implement with your collected data.)
        """
        raise NotImplementedError(
            "Calibrate with your own (command, correct_option) dataset. "
            "See docs/calibration_guide.md for instructions."
        )


# ── VLM wrapper ───────────────────────────────────────────────────────────────

class VLMBackend:
    """
    Wraps a HuggingFace VLM for command interpretation.

    Supported model_ids (swap in config):
      - google/paligemma2-3b-pt-224    (best on Jetson Orin, FP4)
      - HuggingFaceTB/SmolVLM-2B-Instruct
      - Qwen/Qwen2.5-VL-3B-Instruct
      - vikhyatk/moondream2            (lightest, fastest, least capable)
    """

    SYSTEM_PROMPT = """You are the command interpreter for an autonomous outdoor GPS drone.
You receive a natural language command and optionally a camera frame.
Generate exactly 4 candidate action plans (A, B, C, D) plus E="ask for clarification".
Each plan must be a JSON object with keys: action, description, target_alt_m, region_description, constraints.
Then output a JSON object: {"options": {"A": {...}, "B": {...}, "C": {...}, "D": {...}, "E": "ask"}, "probs": {"A": 0.X, "B": 0.X, "C": 0.X, "D": 0.X, "E": 0.X}}
Probabilities must sum to 1.0. Be conservative: if the command is unclear, give E a high probability.
Available actions: fly_to, hover, survey_area, orbit, return, land, follow."""

    def __init__(self, model_id: str, device: str = "cpu", quantize: bool = False):
        self.model_id = model_id
        self.device = device
        self.quantize = quantize
        self._model = None
        self._processor = None
        self._loaded = False

    def _load(self):
        """Lazy-load the model on first inference call."""
        if self._loaded:
            return

        logger.info("Loading VLM: %s (device=%s, quantize=%s)", self.model_id, self.device, self.quantize)

        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            import torch

            kwargs = {"device_map": self.device}
            if self.quantize:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForVision2Seq.from_pretrained(self.model_id, **kwargs)
            self._loaded = True
            logger.info("VLM loaded successfully.")

        except Exception as e:
            logger.error("Failed to load VLM %s: %s", self.model_id, e)
            logger.warning("Falling back to MOCK VLM backend.")
            self._loaded = True  # Mark loaded so we don't retry

    async def interpret(self, text: str, image=None) -> dict:
        """
        Run inference. Returns the raw JSON dict from the VLM.
        Falls back to mock output if model is unavailable.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._interpret_sync, text, image)

    def _interpret_sync(self, text: str, image) -> dict:
        self._load()

        if not self._model:
            return self._mock_interpret(text)

        try:
            import torch
            from PIL import Image as PILImage

            prompt = f"{self.SYSTEM_PROMPT}\n\nCommand: {text}\nResponse:"

            if image is not None:
                pil_image = PILImage.fromarray(image)
                inputs = self._processor(text=prompt, images=pil_image, return_tensors="pt")
            else:
                inputs = self._processor(text=prompt, return_tensors="pt")

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=False,
                    temperature=1.0,
                )

            response = self._processor.decode(outputs[0], skip_special_tokens=True)

            # Extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response[json_start:json_end])

        except Exception as e:
            logger.warning("VLM inference error: %s — using mock", e)

        return self._mock_interpret(text)

    def _mock_interpret(self, text: str) -> dict:
        """
        Deterministic mock for development/testing without a GPU.
        Returns a plausible structured response.
        """
        logger.info("[MOCK VLM] Interpreting: %r", text)

        text_lower = text.lower()

        # Simple keyword heuristics for mock
        if any(w in text_lower for w in ["unclear", "which", "where exactly", "?"]):
            # High ambiguity
            return {
                "options": {
                    "A": {"action": "hover", "description": "Hover in place", "target_alt_m": 20, "region_description": None, "constraints": []},
                    "B": {"action": "fly_to", "description": "Fly north 100m", "target_alt_m": 30, "region_description": "north", "constraints": []},
                    "C": {"action": "survey_area", "description": "Survey nearby area", "target_alt_m": 40, "region_description": "nearby", "constraints": []},
                    "D": {"action": "return", "description": "Return to launch", "target_alt_m": None, "region_description": None, "constraints": []},
                    "E": "ask",
                },
                "probs": {"A": 0.1, "B": 0.15, "C": 0.1, "D": 0.05, "E": 0.6},
            }

        if any(w in text_lower for w in ["return", "home", "rtl", "come back"]):
            return {
                "options": {
                    "A": {"action": "return", "description": "Return to launch point", "target_alt_m": None, "region_description": None, "constraints": []},
                    "B": {"action": "land", "description": "Land immediately", "target_alt_m": 0, "region_description": None, "constraints": []},
                    "C": {"action": "hover", "description": "Hover in place", "target_alt_m": 20, "region_description": None, "constraints": []},
                    "D": {"action": "fly_to", "description": "Fly toward home", "target_alt_m": 20, "region_description": "home", "constraints": []},
                    "E": "ask",
                },
                "probs": {"A": 0.80, "B": 0.10, "C": 0.05, "D": 0.04, "E": 0.01},
            }

        if any(w in text_lower for w in ["hover", "stay", "hold", "wait"]):
            return {
                "options": {
                    "A": {"action": "hover", "description": "Hover at current position", "target_alt_m": 20, "region_description": None, "constraints": []},
                    "B": {"action": "hover", "description": "Hover at 30m altitude", "target_alt_m": 30, "region_description": None, "constraints": []},
                    "C": {"action": "orbit", "description": "Orbit current position", "target_alt_m": 20, "region_description": None, "constraints": []},
                    "D": {"action": "land", "description": "Land here", "target_alt_m": 0, "region_description": None, "constraints": []},
                    "E": "ask",
                },
                "probs": {"A": 0.70, "B": 0.15, "C": 0.10, "D": 0.03, "E": 0.02},
            }

        # Generic flight command — moderate confidence
        return {
            "options": {
                "A": {"action": "fly_to", "description": f"Fly toward described destination", "target_alt_m": 30, "region_description": text, "constraints": ["avoid obstacles"]},
                "B": {"action": "survey_area", "description": "Survey the described area", "target_alt_m": 40, "region_description": text, "constraints": []},
                "C": {"action": "orbit", "description": "Orbit the described point", "target_alt_m": 30, "region_description": text, "constraints": []},
                "D": {"action": "hover", "description": "Hover and observe", "target_alt_m": 25, "region_description": None, "constraints": []},
                "E": "ask",
            },
            "probs": {"A": 0.55, "B": 0.20, "C": 0.12, "D": 0.08, "E": 0.05},
        }


# ── Main interpreter ───────────────────────────────────────────────────────────

class CommandInterpreter:
    """
    Wires VLMBackend + KnowNoGate together.
    """

    DEFAULT_CLARIFICATION_TEMPLATE = (
        "I'm not sure what you mean. Did you want me to:\n"
        "{options}\n"
        "Or something else entirely?"
    )

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.vlm = VLMBackend(
            model_id=cfg.get("model_id", "mock"),
            device=cfg.get("device", "cpu"),
            quantize=cfg.get("quantize", False),
        )
        self.gate = KnowNoGate(
            tau=cfg.get("knowno_tau", 0.6),
            target_coverage=cfg.get("knowno_coverage", 0.85),
        )

    async def interpret(self, text: str, image=None) -> InterpretResult:
        """
        Full interpretation pipeline.
        """
        raw = await self.vlm.interpret(text, image)

        options = raw.get("options", {})
        probs = raw.get("probs", {})

        needs_clarification, pred_set = self.gate.decide(probs)

        if needs_clarification:
            # Build a human-readable clarification question
            # Include all non-E options (even those below tau) as candidates
            option_lines = []
            candidate_keys = sorted(k for k in options if k != "E")
            for key in candidate_keys:
                opt = options.get(key, {})
                desc = opt.get("description", key) if isinstance(opt, dict) else str(opt)
                option_lines.append(f"  {key}) {desc}")

            question = self.DEFAULT_CLARIFICATION_TEMPLATE.format(
                options="\n".join(option_lines) if option_lines else "  (please rephrase your command more specifically)"
            )

            return InterpretResult(
                needs_clarification=True,
                clarification_question=question,
                confidence=max(probs.values()) if probs else 0.0,
                options_considered=list(options.keys()),
            )

        # Singleton — parse the chosen option into a DroneGoal
        chosen_key = pred_set[0]
        chosen = options[chosen_key]

        goal = DroneGoal(
            action=chosen.get("action", "hover"),
            description=chosen.get("description", text),
            target_alt_m=chosen.get("target_alt_m"),
            region_description=chosen.get("region_description"),
            constraints=chosen.get("constraints", []),
            raw_intent=text,
        )

        return InterpretResult(
            needs_clarification=False,
            goal=goal,
            confidence=probs.get(chosen_key, 0.0),
            options_considered=list(options.keys()),
        )
