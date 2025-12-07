"""Tests for Factory Method Selection Guide in Environment class docstring."""

from neurospatial import Environment


def test_environment_docstring_has_factory_selection_guide():
    """Verify that Environment class docstring contains Factory Method Selection Guide."""
    docstring = Environment.__doc__
    assert docstring is not None, "Environment class should have a docstring"

    # Check for "Factory Method Selection" or similar section header
    assert "Factory Method" in docstring or "Choosing a Factory Method" in docstring, (
        "Environment docstring should have a Factory Method Selection section"
    )


def test_factory_guide_mentions_all_six_methods():
    """Verify that the guide references all 6 factory methods."""
    docstring = Environment.__doc__
    assert docstring is not None

    # All factory methods should be mentioned
    factory_methods = [
        "from_samples",
        "from_polygon",
        "from_mask",
        "from_image",
        "from_graph",
        "from_layout",
    ]

    for method in factory_methods:
        assert method in docstring, (
            f"Factory method {method} should be mentioned in the selection guide"
        )


def test_factory_guide_has_use_case_descriptions():
    """Verify that the guide provides use case descriptions."""
    docstring = Environment.__doc__
    assert docstring is not None

    # Should mention common use cases
    use_case_keywords = [
        "position",  # from_samples use case
        "polygon",  # from_polygon use case
        "mask",  # from_mask use case
        "image",  # from_image use case
        "track",  # from_graph use case (linearization)
    ]

    for keyword in use_case_keywords:
        assert keyword in docstring.lower(), (
            f"Use case keyword '{keyword}' should be in the guide"
        )


def test_factory_guide_ordered_by_frequency():
    """Verify that factory methods are ordered by frequency of use."""
    docstring = Environment.__doc__
    assert docstring is not None

    # from_samples should appear before from_layout (most common first)
    samples_pos = docstring.find("from_samples")
    layout_pos = docstring.find("from_layout")

    assert samples_pos != -1 and layout_pos != -1, (
        "Both from_samples and from_layout should be in docstring"
    )
    assert samples_pos < layout_pos, (
        "from_samples should appear before from_layout (ordered by frequency)"
    )


def test_factory_guide_appears_before_attributes():
    """Verify that Factory Method Selection Guide appears before Attributes section."""
    docstring = Environment.__doc__
    assert docstring is not None

    # Find positions
    guide_pos = docstring.find("Factory Method")
    attributes_pos = docstring.find("Attributes\n")

    assert guide_pos != -1, "Factory Method Selection section should exist in docstring"
    assert attributes_pos != -1, "Attributes section should exist in docstring"
    assert guide_pos < attributes_pos, (
        "Factory Method Selection should appear before Attributes"
    )


def test_factory_guide_appears_after_terminology():
    """Verify that Factory Method Selection Guide appears after Terminology section."""
    docstring = Environment.__doc__
    assert docstring is not None

    # Find positions
    terminology_pos = docstring.find("Terminology\n")
    guide_pos = docstring.find("Factory Method")

    assert terminology_pos != -1, "Terminology section should exist in docstring"
    assert guide_pos != -1, "Factory Method Selection section should exist in docstring"
    assert terminology_pos < guide_pos, (
        "Terminology should appear before Factory Method Selection"
    )
