"""Tests for decoding subpackage imports.

These tests verify that the decoding subpackage structure is correctly set up
and all expected public APIs are importable.
"""


class TestPackageImports:
    """Test that package structure is correct."""

    def test_decoding_subpackage_imports(self):
        """Test that decoding subpackage can be imported."""
        import neurospatial.decoding

        # Verify it's a module
        assert hasattr(neurospatial.decoding, "__all__")

    def test_decode_position_importable(self):
        """Test that decode_position can be imported from subpackage."""
        from neurospatial.decoding import decode_position

        assert callable(decode_position)

    def test_decoding_result_importable(self):
        """Test that DecodingResult can be imported from subpackage."""
        from neurospatial.decoding import DecodingResult

        # DecodingResult is a class
        assert isinstance(DecodingResult, type)

    def test_top_level_imports(self):
        """Test that main exports are available from neurospatial."""
        from neurospatial import DecodingResult, decode_position

        assert callable(decode_position)
        assert isinstance(DecodingResult, type)


class TestNormalizeToPosteriorImport:
    """Test that normalize_to_posterior can be imported."""

    def test_normalize_to_posterior_importable(self):
        """Test that normalize_to_posterior can be imported from subpackage."""
        from neurospatial.decoding import normalize_to_posterior

        assert callable(normalize_to_posterior)
