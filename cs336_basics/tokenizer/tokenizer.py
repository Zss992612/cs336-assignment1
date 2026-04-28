from __future__ import annotations

from collections.abc import Iterable, Iterator

import regex as re

from cs336_basics.bpe.train import GPT2_PRETOKEN_PATTERN


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.bytes_to_id = {token: idx for idx, token in self.vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.special_tokens = special_tokens or []
        self.special_token_to_id = {
            token: self.bytes_to_id[token.encode("utf-8")]
            for token in self.special_tokens
            if token.encode("utf-8") in self.bytes_to_id
        }
        self._cache: dict[bytes, list[int]] = {}

        if self.special_tokens:
            alternatives = sorted((re.escape(token) for token in self.special_tokens), key=len, reverse=True)
            self.special_pattern: re.Pattern[str] | None = re.compile("|".join(alternatives))
        else:
            self.special_pattern = None

    def _encode_pretoken(self, token: str) -> list[int]:
        token_bytes = token.encode("utf-8")
        cached = self._cache.get(token_bytes)
        if cached is not None:
            return cached.copy()

        parts = [bytes([byte]) for byte in token_bytes]
        if not parts:
            return []

        while len(parts) > 1:
            ranked_pairs = (
                (self.merge_ranks[pair], pair)
                for pair in zip(parts, parts[1:], strict=False)
                if pair in self.merge_ranks
            )
            best = min(ranked_pairs, default=None)
            if best is None:
                break

            _, pair = best
            merged = pair[0] + pair[1]
            new_parts: list[bytes] = []
            i = 0
            while i < len(parts):
                if i + 1 < len(parts) and parts[i] == pair[0] and parts[i + 1] == pair[1]:
                    new_parts.append(merged)
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts

        ids = [self.bytes_to_id[part] for part in parts]
        self._cache[token_bytes] = ids
        return ids.copy()

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        pos = 0

        if self.special_pattern is None:
            return self._encode_regular_text(text)

        for match in self.special_pattern.finditer(text):
            ids.extend(self._encode_regular_text(text[pos : match.start()]))
            ids.append(self.special_token_to_id[match.group(0)])
            pos = match.end()
        ids.extend(self._encode_regular_text(text[pos:]))
        return ids

    def _encode_regular_text(self, text: str) -> list[int]:
        ids: list[int] = []
        for match in GPT2_PRETOKEN_PATTERN.finditer(text):
            ids.extend(self._encode_pretoken(match.group(0)))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join(self.vocab[idx] for idx in ids)
        return token_bytes.decode("utf-8", errors="replace")
