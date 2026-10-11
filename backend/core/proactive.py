from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class ProactiveEngine:
    """
    Proactive reasoning engine for JARVIS.

    The engine decides whether JARVIS should proactively
    interact with the user.

    It does NOT use command phrases or keyword matching.

    It considers:
        - current context
        - activity
        - user state
        - environment
        - vision
        - screen
        - audio
        - recent conversation
        - previous proactive interaction
        - cooldown
        - priority

    Important design:

        evaluate()
            -> decides whether JARVIS SHOULD speak

        record_proactive_interaction()
            -> records that JARVIS ACTUALLY spoke

    Therefore, merely checking the decision does not start
    the cooldown.
    """

    # =========================================================
    # Configuration
    # =========================================================

    DEFAULT_COOLDOWN_SECONDS = 60

    MIN_PRIORITY = 0.0
    MAX_PRIORITY = 1.0

    DEFAULT_PRIORITY_THRESHOLD = 0.65

    # =========================================================
    # Constructor
    # =========================================================

    def __init__(
        self,
        cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
        priority_threshold: float = DEFAULT_PRIORITY_THRESHOLD,
    ):
        self.cooldown_seconds = max(
            0,
            int(cooldown_seconds),
        )

        self.priority_threshold = max(
            self.MIN_PRIORITY,
            min(
                self.MAX_PRIORITY,
                float(priority_threshold),
            ),
        )

        # Time of the LAST ACTUAL proactive interaction.
        #
        # IMPORTANT:
        # This is NOT updated by evaluate().
        self.last_proactive_at: Optional[datetime] = None

        self.last_reason: Optional[str] = None

        self.last_activity: Optional[str] = None

    # =========================================================
    # Time helpers
    # =========================================================

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    # ---------------------------------------------------------

    def _seconds_since_last_proactive(
        self,
    ) -> Optional[float]:
        """
        Return seconds since the last actual proactive
        interaction.

        Returns None when JARVIS has never proactively
        interacted.
        """

        if self.last_proactive_at is None:
            return None

        elapsed = (
            self._now()
            - self.last_proactive_at
        ).total_seconds()

        return max(
            0.0,
            elapsed,
        )

    # =========================================================
    # Context helpers
    # =========================================================

    @staticmethod
    def _get_activity(
        context: Dict[str, Any],
    ) -> Optional[str]:
        activity = context.get(
            "activity"
        )

        if activity is None:
            return None

        return str(activity).strip() or None

    # ---------------------------------------------------------

    @staticmethod
    def _get_user_state(
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        state = context.get(
            "user_state",
            {},
        )

        if not isinstance(
            state,
            dict,
        ):
            return {}

        return state

    # ---------------------------------------------------------

    @staticmethod
    def _get_environment(
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        environment = context.get(
            "environment",
            {},
        )

        if not isinstance(
            environment,
            dict,
        ):
            return {}

        return environment

    # =========================================================
    # User availability
    # =========================================================

    def _user_available(
        self,
        context: Dict[str, Any],
    ) -> bool:
        """
        Determine whether the user appears available
        for proactive interaction.

        Unknown state is treated as available.

        This allows the proactive system to work even
        before advanced sensors are connected.
        """

        state = self._get_user_state(
            context
        )

        # Explicit unavailable state.
        if state.get(
            "available"
        ) is False:
            return False

        # User is busy.
        if state.get(
            "busy"
        ) is True:
            return False

        # Do not disturb.
        if state.get(
            "do_not_disturb"
        ) is True:
            return False

        return True

    # =========================================================
    # Context change detection
    # =========================================================

    def _activity_changed(
        self,
        activity: Optional[str],
    ) -> bool:
        """
        Determine whether the user's activity changed.

        The first known activity is considered a meaningful
        context event.
        """

        if activity is None:
            return False

        if self.last_activity is None:
            return True

        return activity != self.last_activity

    # =========================================================
    # Recent conversation
    # =========================================================

    @staticmethod
    def _recent_conversation_active(
        history: List[Dict[str, str]],
    ) -> bool:
        """
        Determine whether there is recent conversation
        context.

        This intentionally does not inspect keywords
        or commands.
        """

        if not history:
            return False

        return len(history) > 0

    # =========================================================
    # Candidate generation
    # =========================================================

    def _generate_candidates(
        self,
        context: Dict[str, Any],
        history: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """
        Generate possible reasons for proactive interaction.

        These are context signals, not commands.
        """

        candidates: List[Dict[str, Any]] = []

        activity = self._get_activity(
            context
        )

        user_state = self._get_user_state(
            context
        )

        environment = self._get_environment(
            context
        )

        # -----------------------------------------------------
        # Activity changed
        # -----------------------------------------------------

        if self._activity_changed(
            activity
        ):
            candidates.append(
                {
                    "reason": "activity_changed",
                    "priority": 0.72,
                    "context": {
                        "activity": activity,
                    },
                }
            )

        # -----------------------------------------------------
        # User became available
        # -----------------------------------------------------

        if (
            user_state.get(
                "available"
            )
            is True
        ):
            candidates.append(
                {
                    "reason": "user_available",
                    "priority": 0.68,
                    "context": {},
                }
            )

        # -----------------------------------------------------
        # Environment changed
        # -----------------------------------------------------

        if (
            environment.get(
                "changed"
            )
            is True
        ):
            candidates.append(
                {
                    "reason": "environment_changed",
                    "priority": 0.70,
                    "context": environment,
                }
            )

        # -----------------------------------------------------
        # Vision detected meaningful change
        # -----------------------------------------------------

        vision = context.get(
            "vision",
            {},
        )

        if isinstance(
            vision,
            dict,
        ):
            if (
                vision.get(
                    "significant_change"
                )
                is True
            ):
                candidates.append(
                    {
                        "reason": "visual_context_changed",
                        "priority": 0.75,
                        "context": vision,
                    }
                )

        # -----------------------------------------------------
        # Screen changed
        # -----------------------------------------------------

        screen = context.get(
            "screen",
            {},
        )

        if isinstance(
            screen,
            dict,
        ):
            if (
                screen.get(
                    "significant_change"
                )
                is True
            ):
                candidates.append(
                    {
                        "reason": "screen_context_changed",
                        "priority": 0.73,
                        "context": screen,
                    }
                )

        # -----------------------------------------------------
        # Audio/environment event
        # -----------------------------------------------------

        audio = context.get(
            "audio",
            {},
        )

        if isinstance(
            audio,
            dict,
        ):
            if (
                audio.get(
                    "significant_event"
                )
                is True
            ):
                candidates.append(
                    {
                        "reason": "audio_context_changed",
                        "priority": 0.70,
                        "context": audio,
                    }
                )

        # -----------------------------------------------------
        # Recent conversation
        # -----------------------------------------------------

        if self._recent_conversation_active(
            history
        ):
            candidates.append(
                {
                    "reason": "conversation_context",
                    "priority": 0.66,
                    "context": {},
                }
            )

        return candidates

    # =========================================================
    # Candidate selection
    # =========================================================

    @staticmethod
    def _select_candidate(
        candidates: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Select the strongest proactive reason.
        """

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda item: float(
                item.get(
                    "priority",
                    0.0,
                )
            ),
        )

    # =========================================================
    # Evaluate
    # =========================================================

    def evaluate(
        self,
        context: Dict[str, Any],
        history: Optional[
            List[Dict[str, str]]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate whether JARVIS should proactively interact.

        IMPORTANT:

        This method ONLY makes a decision.

        It does NOT record a proactive interaction.

        The cooldown starts only when
        record_proactive_interaction() is called after
        JARVIS actually generates/sends a response.
        """

        history = history or []

        # -----------------------------------------------------
        # Validate context
        # -----------------------------------------------------

        if not isinstance(
            context,
            dict,
        ):
            context = {}

        # -----------------------------------------------------
        # User unavailable
        # -----------------------------------------------------

        if not self._user_available(
            context
        ):
            return {
                "should_speak": False,
                "decision": "wait",
                "reason": "user_unavailable",
                "priority": 0.0,
                "cooldown_active": False,
            }

        # -----------------------------------------------------
        # Cooldown
        # -----------------------------------------------------

        elapsed = (
            self._seconds_since_last_proactive()
        )

        if (
            elapsed is not None
            and elapsed < self.cooldown_seconds
        ):
            remaining = (
                self.cooldown_seconds
                - elapsed
            )

            return {
                "should_speak": False,
                "decision": "cooldown",
                "reason": "recent_proactive_interaction",
                "priority": 0.0,
                "cooldown_active": True,
                "cooldown_remaining": round(
                    max(
                        0.0,
                        remaining,
                    ),
                    2,
                ),
            }

        # -----------------------------------------------------
        # Generate candidates
        # -----------------------------------------------------

        candidates = self._generate_candidates(
            context=context,
            history=history,
        )

        # -----------------------------------------------------
        # No reason to speak
        # -----------------------------------------------------

        if not candidates:
            return {
                "should_speak": False,
                "decision": "wait",
                "reason": "nothing_relevant",
                "priority": 0.0,
                "cooldown_active": False,
            }

        # -----------------------------------------------------
        # Select strongest reason
        # -----------------------------------------------------

        candidate = self._select_candidate(
            candidates
        )

        if candidate is None:
            return {
                "should_speak": False,
                "decision": "wait",
                "reason": "no_candidate",
                "priority": 0.0,
                "cooldown_active": False,
            }

        priority = float(
            candidate.get(
                "priority",
                0.0,
            )
        )

        reason = candidate.get(
            "reason",
            "unknown",
        )

        # -----------------------------------------------------
        # Priority threshold
        # -----------------------------------------------------

        if priority < self.priority_threshold:
            return {
                "should_speak": False,
                "decision": "wait",
                "reason": "priority_too_low",
                "candidate_reason": reason,
                "priority": priority,
                "threshold": self.priority_threshold,
                "cooldown_active": False,
            }

        # -----------------------------------------------------
        # Allow proactive interaction
        # -----------------------------------------------------

        # IMPORTANT:
        #
        # Do NOT update last_proactive_at here.
        #
        # evaluate() only says:
        #
        # "JARVIS is allowed to speak."
        #
        # The actual interaction is recorded later by:
        #
        # record_proactive_interaction()

        return {
            "should_speak": True,
            "decision": "speak",
            "reason": reason,
            "priority": priority,
            "threshold": self.priority_threshold,
            "cooldown_active": False,
            "context": candidate.get(
                "context",
                {},
            ),
        }

    # =========================================================
    # Record actual proactive interaction
    # =========================================================

    def record_proactive_interaction(
        self,
        reason: Optional[str] = None,
        activity: Optional[str] = None,
    ) -> None:
        """
        Record that JARVIS actually performed a proactive
        interaction.

        This is where the cooldown starts.

        Call this ONLY after the proactive response has been
        successfully generated/sent.
        """

        self.last_proactive_at = self._now()

        if reason is not None:
            self.last_reason = reason

        if activity is not None:
            self.last_activity = activity

    # =========================================================
    # Reset
    # =========================================================

    def reset(self) -> None:
        """
        Reset proactive state.
        """

        self.last_proactive_at = None
        self.last_reason = None
        self.last_activity = None

    # =========================================================
    # Status
    # =========================================================

    def status(self) -> Dict[str, Any]:
        """
        Return proactive engine state.
        """

        elapsed = (
            self._seconds_since_last_proactive()
        )

        cooldown_active = (
            elapsed is not None
            and elapsed < self.cooldown_seconds
        )

        cooldown_remaining = 0.0

        if cooldown_active:
            cooldown_remaining = max(
                0.0,
                self.cooldown_seconds - elapsed,
            )

        return {
            "enabled": True,
            "cooldown_seconds": self.cooldown_seconds,
            "priority_threshold": self.priority_threshold,
            "last_reason": self.last_reason,
            "last_activity": self.last_activity,
            "seconds_since_last_proactive": elapsed,
            "cooldown_active": cooldown_active,
            "cooldown_remaining": round(
                cooldown_remaining,
                2,
            ),
        }