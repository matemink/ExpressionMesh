from enum import IntEnum


class Expression(IntEnum):
    """Stable class identifiers used by both datasets and serialized models."""

    HAPPY = 0
    SAD = 1
    SURPRISED = 2

    @property
    def directory_name(self) -> str:
        return self.name.lower()


EXPRESSIONS = tuple(Expression)
EXPECTED_CLASS_IDS = tuple(expression.value for expression in EXPRESSIONS)


def expression_from_id(class_id: int | float) -> Expression:
    numeric_id = int(class_id)
    if numeric_id != class_id:
        raise ValueError(f"Expression class must be an integer, got {class_id!r}")

    try:
        return Expression(numeric_id)
    except ValueError as error:
        raise ValueError(f"Unknown expression class: {numeric_id}") from error
