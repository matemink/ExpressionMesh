import pytest

from expression_mesh.labels import Expression, expression_from_id


def test_stable_class_mapping() -> None:
    assert [expression.name for expression in Expression] == ["HAPPY", "SAD", "SURPRISED"]
    assert expression_from_id(2) is Expression.SURPRISED


def test_unknown_class_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown expression class"):
        expression_from_id(3)
