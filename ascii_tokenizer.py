# ascii_tokenizer.py
class SimpleASCIITokenizer:
    """ A simple tokenizer for the first 256 ASCII characters. """
    def __init__(self):
        self.chars = [chr(i) for i in range(256)]
        self.vocab_size = len(self.chars)
        self.char_to_int = {ch: i for i, ch in enumerate(self.chars)}
        self.int_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> list[int]:
        """Converts a string to a list of integer token IDs."""
        return [self.char_to_int.get(char, 0) for char in text]

    def decode(self, tokens: list[int]) -> str:
        """Converts a list of integer token IDs back to a string."""
        return "".join([self.int_to_char.get(token, '') for token in tokens])