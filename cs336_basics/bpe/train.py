from __future__ import annotations

import os
from collections import Counter, defaultdict

import regex as re


GPT2_PRETOKEN_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _special_token_pattern(special_tokens: list[str]) -> re.Pattern[str] | None:
    if not special_tokens:
        return None
    alternatives = sorted((re.escape(token) for token in special_tokens), key=len, reverse=True)
    return re.compile("|".join(alternatives))


def _pretoken_counts(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    special_pattern = _special_token_pattern(special_tokens)
    counts: Counter[tuple[bytes, ...]] = Counter()

    chunks = special_pattern.split(text) if special_pattern is not None else [text]
    for chunk in chunks:
        for match in GPT2_PRETOKEN_PATTERN.finditer(chunk):
            token_bytes = match.group(0).encode("utf-8")
            counts[tuple(bytes([byte]) for byte in token_bytes)] += 1
    return counts


def _word_pair_counts(word: tuple[bytes, ...]) -> Counter[tuple[bytes, bytes]]:
    return Counter(zip(word, word[1:], strict=False))


def _merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes], merged: bytes) -> tuple[bytes, ...]:
    new_word: list[bytes] = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]:
            new_word.append(merged)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    special_tokens = special_tokens or []

    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    words = _pretoken_counts(text, special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    word_list = list(words.keys())
    freqs = [words[word] for word in word_list]
    counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_word_ids: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    for word_id, word in enumerate(word_list):
        freq = freqs[word_id]
        for pair, pair_count in _word_pair_counts(word).items():
            counts[pair] += pair_count * freq
            pair_to_word_ids[pair].add(word_id)

    while len(vocab) < vocab_size:
        if not counts:
            break

        best_pair = max(counts, key=lambda pair: (counts[pair], pair))
        merged = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocab[len(vocab)] = merged

        affected_word_ids = list(pair_to_word_ids.pop(best_pair, set()))
        counts.pop(best_pair, None)

        for word_id in affected_word_ids:
            old_word = word_list[word_id]
            if best_pair not in _word_pair_counts(old_word):
                continue

            freq = freqs[word_id]
            for pair, pair_count in _word_pair_counts(old_word).items():
                counts[pair] -= pair_count * freq
                if counts[pair] <= 0:
                    counts.pop(pair, None)

            new_word = _merge_word(old_word, best_pair, merged)
            word_list[word_id] = new_word

            for pair, pair_count in _word_pair_counts(new_word).items():
                counts[pair] += pair_count * freq
                pair_to_word_ids[pair].add(word_id)

    return vocab, merges
