from core.proactive import ProactiveEngine


def check(
    condition,
    message,
):
    if not condition:
        raise AssertionError(
            message
        )

    print(
        f"[PASS] {message}"
    )


def main():

    print("=" * 60)
    print("JARVIS PROACTIVE REASONING TEST")
    print("=" * 60)

    engine = ProactiveEngine(
        cooldown_seconds=60,
        priority_threshold=0.65,
    )

    # =========================================================
    # 1. Empty context
    # =========================================================

    print(
        "\nTesting empty context..."
    )

    result = engine.evaluate(
        context={},
        history=[],
    )

    print(result)

    check(
        result["should_speak"]
        is False,
        "JARVIS waits when there is nothing relevant",
    )

    # =========================================================
    # 2. Activity change
    # =========================================================

    print(
        "\nTesting activity change..."
    )

    engine.reset()

    result = engine.evaluate(
        context={
            "activity": "working on JARVIS",
        },
        history=[],
    )

    print(result)

    check(
        result["should_speak"]
        is True,
        "JARVIS can detect a meaningful activity change",
    )

    check(
        result["decision"]
        == "speak",
        "Proactive decision is speak",
    )

    check(
        result["reason"]
        == "activity_changed",
        "Activity change is identified as the reason",
    )

    # =========================================================
    # 3. Cooldown
    # =========================================================

    print(
        "\nTesting cooldown..."
    )

    result = engine.evaluate(
        context={
            "activity": "working on JARVIS",
        },
        history=[],
    )

    print(result)

    check(
        result["should_speak"]
        is False,
        "JARVIS does not immediately interrupt again",
    )

    check(
        result["cooldown_active"]
        is True,
        "Proactive cooldown is active",
    )

    # =========================================================
    # 4. User unavailable
    # =========================================================

    print(
        "\nTesting user availability..."
    )

    engine.reset()

    result = engine.evaluate(
        context={
            "activity": "working",
            "user_state": {
                "available": False,
            },
        },
        history=[],
    )

    print(result)

    check(
        result["should_speak"]
        is False,
        "JARVIS stays silent when user is unavailable",
    )

    check(
        result["reason"]
        == "user_unavailable",
        "Unavailable user state is detected",
    )

    # =========================================================
    # 5. Environment change
    # =========================================================

    print(
        "\nTesting environment change..."
    )

    engine.reset()

    result = engine.evaluate(
        context={
            "environment": {
                "changed": True,
            },
        },
        history=[],
    )

    print(result)

    check(
        result["should_speak"]
        is True,
        "JARVIS can react to meaningful environment changes",
    )

    # =========================================================
    # 6. Vision change
    # =========================================================

    print(
        "\nTesting visual context..."
    )

    engine.reset()

    result = engine.evaluate(
        context={
            "vision": {
                "significant_change": True,
            },
        },
        history=[],
    )

    print(result)

    check(
        result["should_speak"]
        is True,
        "JARVIS can react to significant visual changes",
    )

    # =========================================================
    # 7. Screen change
    # =========================================================

    print(
        "\nTesting screen context..."
    )

    engine.reset()

    result = engine.evaluate(
        context={
            "screen": {
                "significant_change": True,
            },
        },
        history=[],
    )

    print(result)

    check(
        result["should_speak"]
        is True,
        "JARVIS can react to significant screen changes",
    )

    # =========================================================
    # Status
    # =========================================================

    print(
        "\nProactive engine status:"
    )

    print(
        engine.status()
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "PROACTIVE REASONING TEST PASSED"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()