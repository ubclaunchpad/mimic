"""Tests for util class."""
import unittest
import mimic.util as util


class TestUtil(unittest.TestCase):
    """Tests for util class."""

    def test_clean_string_basic(self):
        """Test basic string cleaning."""
        cleaned_txt = util.clean_text("hello ,Ʈ asdf!!!!!!")
        assert(cleaned_txt == "hello asdf")

    def test_clean_string_emoji(self):
        """Test cleaning of emojis and punctuation."""
        cleaned_txt = util.clean_text("""😁😁😁 , this is a tEStinG
                                      STRING!!!! | 😁 how u doin\'""")
        assert(cleaned_txt == "this is a testing string how u doin")

    def test_clean_string_shakespeare(self):
        """Test cleaning of whitespace."""
        cleaned_txt = util.clean_text("""Upon my sword for ever:
                                      'Agripo'er, his days let me free.
                                      \nStop it of that word, be so:
                                      at Lear,\tWhen I did profess the hour
                                      -stranger for my life,""")
        assert(cleaned_txt == """upon my sword for ever agr
                                 ipo er his days let me free sto
                                 p it of that word be so at lear wh
                                 en i did profess the hour stranger f
                                 or my life""")

    def test_input_word_length_not_met(self):
        """Test minimum word requirement not met."""
        with self.assertRaises(ValueError):
            util.verify_text("i " * 19999)

    def test_input_word_length_met(self):
        """Test minimum word requirement met."""
        util.verify_text("i " * 20000)
