"""Constrained decoding prototype (DECISIONS.md B6, plan C): the teacher may only emit whole
original words, in original order, each at most once per occurrence.

Works on bytes, because the teacher tokenizer is byte-level BPE (a CJK character can span tokens).
Output grammar, matching align.py's boundary rules:
    output   := word (sep word)*  END
    sep      := "" | " "          (one optional space; never inside a word)
    a space is required between kept words k < i when the original has whitespace between them
    and both sides are letters/digits (otherwise they would merge, e.g. "a part" -> "apart").
The model chooses which words to keep and when to stop; the constraint only removes illegal tokens.
The final text is still checked independently by align().
"""
from bisect import bisect_left
from functools import lru_cache

SPACE = 0x20


class TokenTable:
    """Byte strings of the teacher vocabulary, sorted for prefix search."""

    def __init__(self, id_bytes, kind="byte-level"):
        # id_bytes[i] = bytes of token i, or None for special/added tokens
        self.id_bytes = id_bytes
        self.kind = kind
        pairs = sorted((b, i) for i, b in enumerate(id_bytes) if b)
        self.keys = [b for b, _ in pairs]
        self.ids = [i for _, i in pairs]

    def has(self, b):
        j = bisect_left(self.keys, b)
        return j < len(self.keys) and self.keys[j] == b

    @classmethod
    def from_hf(cls, tok):
        """Byte-level BPE (GPT-2 alphabet) or SentencePiece ("▁" = space, no byte fallback)."""
        import json
        decoder = (json.loads(tok.backend_tokenizer.to_str()).get("decoder") or {}).get("type")
        special = set(tok.all_special_ids) | set(tok.added_tokens_decoder)
        pieces = tok.convert_ids_to_tokens(list(range(len(tok))))
        if decoder == "Metaspace":
            return cls([None if i in special or t is None else t.replace("▁", " ").encode("utf-8")
                        for i, t in enumerate(pieces)], kind="sentencepiece")
        # GPT-2 byte-level alphabet: each byte is shown as one printable unicode character
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs, n = bs[:], 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        decode = {chr(c): b for b, c in zip(bs, cs)}
        id_bytes = []
        for i, t in enumerate(pieces):
            if i in special or t is None or any(ch not in decode for ch in t):
                id_bytes.append(None)
            else:
                id_bytes.append(bytes(decode[ch] for ch in t))
        return cls(id_bytes, kind="byte-level")


class DeletionConstraint:
    """States: ("B", k, sp) = at a boundary after kept word k (-1 = nothing yet), sp = space emitted;
               ("W", i, off) = inside word i, next byte is words[i][off]."""

    def __init__(self, text, spans, table, end_response_id, eos_id, allow_leading_space=False):
        # allow_leading_space: SentencePiece pieces start with "▁", so the first token is usually
        # " word"; the decoder drops that space. Byte-level BPE keeps the original grammar (False).
        self.table = table
        self.allow_leading_space = allow_leading_space
        self.end_response_id, self.eos_id = end_response_id, eos_id
        idx = [j for j, (s, e) in enumerate(spans) if e > s]  # content words only
        self.words = [text[spans[j][0]:spans[j][1]].encode("utf-8") for j in idx]
        starts = [spans[j][0] for j in idx]
        ends = [spans[j][1] for j in idx]
        first_alnum = [text[spans[j][0]].isalnum() for j in idx]
        last_alnum = [text[spans[j][1] - 1].isalnum() for j in idx]
        cws = [0]
        for c in text:
            cws.append(cws[-1] + c.isspace())
        n = len(self.words)

        def need_space(k, i):
            return (k >= 0 and last_alnum[k] and first_alnum[i]
                    and cws[starts[i]] - cws[ends[k]] > 0)

        @lru_cache(maxsize=None)
        def next_words(k, sp):
            # first byte -> later words that may start here
            table_ = {}
            for i in range(k + 1, n):
                if sp or not need_space(k, i):
                    table_.setdefault(self.words[i][0], []).append(i)
            return table_

        self._next_words = next_words

        @lru_cache(maxsize=None)
        def finishable(rest):
            # rest (bytes still to write in a word) can be spelled exactly by tokens from a token
            # boundary. Sufficient, not necessary: tokens crossing into the next word are still
            # allowed by the byte-level search, they are just not relied on to finish a word.
            if not rest or table.has(rest):
                return True
            return any(table.has(rest[:k]) and finishable(rest[k:]) for k in range(1, len(rest)))

        self._finishable = finishable
        # words no token sequence can spell (e.g. "，" when the vocabulary normalises it to ","):
        # they can only be dropped; counted for the report
        self.unreachable_words = sum(not (finishable(w) or finishable(b" " + w)) for w in self.words)
        self.states = frozenset([("B", -1, 0)])
        self.ended = False      # <|END_RESPONSE|> emitted; only EOS may follow
        self.consumed = 0       # generated tokens already fed
        self.no_legal_path = False

    # --- automaton -------------------------------------------------------
    def _bytes_from(self, states):
        out = set()
        for st in states:
            if st[0] == "W":
                out.add(self.words[st[1]][st[2]])
            else:
                _, k, sp = st
                if (k >= 0 or self.allow_leading_space) and not sp:
                    out.add(SPACE)
                out.update(self._next_words(k, sp))
        return out

    def _step(self, states, b):
        nxt = set()
        for st in states:
            if st[0] == "W":
                _, i, off = st
                if self.words[i][off] == b:
                    nxt.add(("B", i, 0) if off + 1 == len(self.words[i]) else ("W", i, off + 1))
            else:
                _, k, sp = st
                if b == SPACE and (k >= 0 or self.allow_leading_space) and not sp:
                    nxt.add(("B", k, 1))
                for i in self._next_words(k, sp).get(b, ()):
                    nxt.add(("B", i, 0) if len(self.words[i]) == 1 else ("W", i, 1))
        return frozenset(nxt)

    def _viable(self, states):
        """Keep states from which the output can still be completed at a token boundary."""
        return frozenset(st for st in states
                         if st[0] == "B" or self._finishable(self.words[st[1]][st[2]:]))

    def can_end(self, states=None):
        return any(st[0] == "B" and st[1] >= 0 for st in (states or self.states))

    # --- token level -----------------------------------------------------
    def allowed_tokens(self):
        if self.ended:
            return [self.eos_id]
        keys, ids, out = self.table.keys, self.table.ids, []

        def dfs(prefix, lo, hi, states):
            for b in self._bytes_from(states):
                sub = prefix + bytes([b])
                l = bisect_left(keys, sub, lo, hi)
                h = bisect_left(keys, prefix + bytes([b + 1]), l, hi) if b < 255 else hi
                if l == h:
                    continue
                ns = self._step(states, b)
                if not ns:
                    continue
                j = l
                ok_here = None
                while j < h and keys[j] == sub:  # tokens exactly equal to sub
                    if ok_here is None:
                        ok_here = bool(self._viable(ns))  # never enter a word that cannot be finished
                    if ok_here:
                        out.append(ids[j])
                    j += 1
                if j < h:
                    dfs(sub, j, h, ns)

        dfs(b"", 0, len(keys), self.states)
        if self.can_end():
            out.append(self.end_response_id)
        return out

    def feed(self, token_id):
        if token_id == self.end_response_id:
            self.ended = True
            return
        if self.ended:
            return
        bs = self.table.id_bytes[token_id]
        states = self.states
        for b in bs or b"":
            states = self._step(states, b)
        states = self._viable(states)
        if not states:  # a token outside the grammar got through; record, never patch
            self.no_legal_path = True
            return
        self.states = states

    def prefix_allowed_tokens_fn(self, prompt_len):
        """For HF generate(prefix_allowed_tokens_fn=...). Batch size 1."""
        def fn(batch_id, input_ids):
            gen = input_ids[prompt_len:].tolist()
            for t in gen[self.consumed:]:
                self.feed(t)
            self.consumed = len(gen)
            allowed = self.allowed_tokens()
            if not allowed:  # cannot happen with a full byte-level vocab; recorded, never patched
                self.no_legal_path = True
                return [self.eos_id]
            return allowed
        return fn
