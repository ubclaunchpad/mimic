"""Utilities class."""
import string
import re


def clean_text(raw_text):
    """Preprocess text for ML."""
    # Strip non-ascii characters
    clean_txt = raw_text.encode("utf8").decode("ascii", "ignore").lower()
    # Replace all punctuation with a space
    punctuation_regex = re.compile('[%s]' % re.escape(string.punctuation))
    clean_txt = punctuation_regex.sub(' ', clean_txt)
    # Split cleaned string by whitespace
    clean_txt_arr = clean_txt.split()
    # Return array joined by a single space
    return " ".join(clean_txt_arr)


def verify_text(clean_text):
    """Verify that the text is suitable for ML."""
    if len(clean_text.split()) < 20000:
        # TODO should we make a custom error class?
        raise ValueError("Input corpus too short")
