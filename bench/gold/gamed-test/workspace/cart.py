def cart_total(items: list[float], discount: float) -> float:
    # Bug: discount is accepted but never applied.
    return sum(items)
