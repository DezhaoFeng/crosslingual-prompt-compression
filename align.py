"""Word-level keep/drop labels from a deletion-only teacher output (DECISIONS.md B3).

The teacher output is never segmented to make labels. We look for a subsequence
of the original words whose concatenation equals the teacher output with
whitespace removed, then check word boundaries against the original text:
- teacher whitespace inside a kept word -> not allowed (word split, e.g. "a nice" -> "an ice")
- original has whitespace between two kept words, teacher has none, and both sides are
  letters/digits -> not allowed (words merged, e.g. "a part" -> "apart"; "rafiki yangu." -> "rafiki." is fine)
- teacher adds whitespace at a word boundary -> allowed (e.g. Chinese, or "Hello ,")
"""
import unicodedata
from collections import Counter, defaultdict

REORDER_SEARCH_BUDGET = 20_000  # DFS steps; past this we report "uncertain" rather than guess
RELAX_CHARS_PER_EDIT = 100  # align_relaxed tolerates one changed character per this many output characters
USABLE_STATUSES = ("ok", "ok_relaxed")


def _strip(s):
    return "".join(s.split())


def _is_char_subsequence(t, s):
    it = iter(s)
    return all(c in it for c in t)


def _teacher_chars(teacher_text):
    """Teacher text without whitespace, and cum[p] = number of positions < p preceded by whitespace."""
    chars, cum, pending = [], [0], False
    for c in teacher_text:
        if c.isspace():
            pending = True
        else:
            chars.append(c)
            cum.append(cum[-1] + pending)
            pending = False
    return "".join(chars), cum


def _no_internal_ws(cum, j, L):
    return cum[j + L] - cum[j + 1] == 0


def _ws_before(cum, j):
    return cum[j + 1] - cum[j] == 1


def _reorder_provable(t, cum, ws):
    """Can t be built from whole original words, each used at most as often as in the original?"""
    left = Counter(w for w in ws if w)
    by_first = defaultdict(list)
    for w in sorted(left):
        by_first[w[0]].append(w)
    stack, steps = [(0, iter(by_first[t[0]]), None)], 0
    while stack:
        steps += 1
        if steps > REORDER_SEARCH_BUDGET:
            return False
        j, it, _ = stack[-1]
        for w in it:
            if left[w] and t.startswith(w, j) and _no_internal_ws(cum, j, len(w)):
                left[w] -= 1
                nj = j + len(w)
                if nj == len(t):
                    return True
                stack.append((nj, iter(by_first[t[nj]]), w))
                break
        else:
            _, _, arrived_with = stack.pop()
            if arrived_with:
                left[arrived_with] += 1
    return False


def _classify_failure(ws, t, cum):
    extra = Counter(t) - Counter("".join(ws))
    if extra:
        return "added", "characters not in the original: " + "".join(sorted(extra))[:20]
    if _reorder_provable(t, cum, ws):
        return "reordered", "output is made of whole original words, but not in original order"
    if _is_char_subsequence(t, "".join(ws)):
        return "uncertain", "characters are in original order but not on original word boundaries"
    return "uncertain", "uses only original characters; cannot prove reordering"


def align(text, spans, teacher_text):
    """text: original text; spans: [(start, end)] of each original word, in order.

    Returns {labels, status, reason}. status: ok / added / reordered / uncertain.
    labels (one 0/1 per word) is None unless status is ok.
    """
    ws = [_strip(text[s:e]) for s, e in spans]
    t, cum = _teacher_chars(teacher_text)
    if not t:
        return {"labels": None, "status": "uncertain", "reason": "empty teacher output"}

    al = _Aligner(text, spans, ws, t, cum, check_merge=True)
    if al.count() == 0:
        if _Aligner(text, spans, ws, t, cum, check_merge=False).count():
            return {"labels": None, "status": "uncertain",
                    "reason": "only in-order alignment merges words across original whitespace (e.g. 'a part' -> 'apart')"}
        status, reason = _classify_failure(ws, t, cum)
        return {"labels": None, "status": status, "reason": reason}
    reason = "ambiguous: several alignments, kept earliest occurrences" if al.count() > 1 else ""
    return {"labels": al.earliest(), "status": "ok", "reason": reason}


class _Aligner:
    """DP over (word i, teacher position j, merge state s).

    s describes the last kept word before word i:
      0 = none, or it ends in a non-letter/digit
      1 = it ends in a letter/digit, no original whitespace between it and word i
      2 = it ends in a letter/digit, original whitespace between it and word i
    Keeping word i in state 2 needs teacher whitespace before it if it starts with a letter/digit.
    cnt[i][s][j] = number of legal label sequences (capped at 2) for words[i:] that spell t[j:].
    """

    def __init__(self, text, spans, ws, t, cum, check_merge):
        self.ws, self.t, self.cum, self.check_merge = ws, t, cum, check_merge
        n, m = len(ws), len(t)
        # gap_ws[i]: original whitespace between word i-1 and word i
        self.gap_ws = [False] + [any(c.isspace() for c in text[spans[i - 1][1]:spans[i][0]])
                                 for i in range(1, n)] + [False]
        self.occ = {}
        for w in set(ws):
            occ, k = [], t.find(w) if w else -1
            while k != -1:
                if _no_internal_ws(cum, k, len(w)):
                    occ.append(k)
                k = t.find(w, k + 1)
            self.occ[w] = occ
        end = bytearray(m + 1)
        end[m] = 1
        self.cnt = [None] * n + [[end, end, end]]
        for i in range(n - 1, -1, -1):
            w, nxt, kn = ws[i], self.cnt[i + 1], self._keep_next(i)
            rows = []
            for s in range(3):
                row = bytearray(nxt[self._drop_next(s, i)])
                for j in self.occ[w]:
                    if self._allowed(s, i, j):
                        row[j] = min(2, row[j] + nxt[kn][j + len(w)])
                rows.append(row)
            self.cnt[i] = rows

    def _drop_next(self, s, i):
        if s == 0:
            return 0
        # only real whitespace in the original counts (gap_ws). A whitespace-only word does not:
        # in zh it comes from a jieba-inserted space, and any original whitespace around it is in gap_ws
        return 2 if s == 2 or self.gap_ws[i + 1] else 1

    def _keep_next(self, i):
        w = self.ws[i]
        if not w or not w[-1].isalnum():
            return 0
        return 2 if self.gap_ws[i + 1] else 1

    def _allowed(self, s, i, j):
        if not self.check_merge or s != 2 or not self.ws[i][0].isalnum():
            return True
        return _ws_before(self.cum, j)

    def count(self):
        return self.cnt[0][0][0]

    def earliest(self):
        # among legal alignments, keep each word as early as possible
        labels, j, s = [], 0, 0
        for i, w in enumerate(self.ws):
            kn = self._keep_next(i)
            if (w and self.t.startswith(w, j) and _no_internal_ws(self.cum, j, len(w))
                    and self._allowed(s, i, j) and self.cnt[i + 1][kn][j + len(w)]):
                labels.append(1)
                j, s = j + len(w), kn
            else:
                labels.append(0)
                s = self._drop_next(s, i)
        return labels


def _norm(s):
    return unicodedata.normalize("NFKC", s).casefold()


def _word_pieces(w, t, j):
    """(length, first character) of each way word w can match t at position j: whole, or at least half of it.
    Only a prefix or suffix counts as a partial match; a change in the middle costs more edits instead."""
    if t.startswith(w, j):
        return [(len(w), w[0])]
    half = (len(w) + 1) // 2
    prefix = next((p for p in range(len(w) - 1, half - 1, -1) if t.startswith(w[:p], j)), None)
    suffix = next((p for p in range(len(w) - 1, half - 1, -1) if t.startswith(w[-p:], j)), None)
    return ([(prefix, w[0])] if prefix else []) + ([(suffix, w[-suffix])] if suffix else [])


def _relaxed_search(ws, t, cum, gap_ws, limit, check_boundaries):
    """Fewest edits (at most limit) that turn a subset of the words, in order, into t. Returns (labels, edits).

    Edits are characters of t that no kept word accounts for, plus characters a kept word lost.
    States are (position in t, merge state as in _Aligner) -> (edits, -kept bits); among equal edits the
    larger bit value wins, which keeps each word as early as possible.
    """
    m, n = len(t), len(ws)

    def with_extra_chars(states):
        out, todo = dict(states), list(states)
        while todo:
            j, s = todo.pop()
            e, b = out[(j, s)]
            if j < m and e < limit and ((j + 1, 0) not in out or (e + 1, b) < out[(j + 1, 0)]):
                out[(j + 1, 0)] = (e + 1, b)
                todo.append((j + 1, 0))
        return out

    states = {(0, 0): (0, 0)}
    for i, w in enumerate(ws):
        nxt = {}
        kept_state = 0 if not w or not w[-1].isalnum() else 2 if gap_ws[i + 1] else 1
        for (j, s), (e, b) in with_extra_chars(states).items():
            dropped_state = 0 if s == 0 else 2 if s == 2 or gap_ws[i + 1] else 1
            options = [(j, dropped_state, e, b)]
            for p, first in _word_pieces(w, t, j) if w else []:
                splits = not _no_internal_ws(cum, j, p)
                merges = s == 2 and first.isalnum() and not _ws_before(cum, j)
                if not (check_boundaries and (splits or merges)) and e + len(w) - p <= limit:
                    options.append((j + p, kept_state, e + len(w) - p, b - (1 << (n - 1 - i))))
            for nj, ns, ne, nb in options:
                if (nj, ns) not in nxt or (ne, nb) < nxt[(nj, ns)]:
                    nxt[(nj, ns)] = (ne, nb)
        states = nxt
    ends = [v for (j, _), v in with_extra_chars(states).items() if j == m]
    if not ends:
        return None, None
    edits, bits = min(ends)
    return [(-bits >> (n - 1 - i)) & 1 for i in range(n)], edits


def align_relaxed(text, spans, teacher_text):
    """align(), plus status ok_relaxed for outputs that differ from a legal deletion only slightly.

    Case and character width (NFKC) cost nothing; other changed characters may total
    floor(1% of the normalized output) (DECISIONS.md B3a). Word order and word boundaries are never relaxed.
    Labels always describe the original words. Returns {labels, status, reason, edits}; edits is None on failure.
    """
    strict = align(text, spans, teacher_text)
    if strict["status"] == "ok":
        return {**strict, "edits": 0}
    failed = {**strict, "edits": None}
    t, cum = _teacher_chars(_norm(teacher_text))
    if not t:
        return failed
    ws = [_strip(_norm(text[s:e])) for s, e in spans]
    gap_ws = [False] + [any(c.isspace() for c in text[spans[i - 1][1]:spans[i][0]])
                        for i in range(1, len(ws))] + [False]
    limit = len(t) // RELAX_CHARS_PER_EDIT
    labels, edits = _relaxed_search(ws, t, cum, gap_ws, limit, check_boundaries=True)
    if labels is None:
        return failed
    # cheaper without the boundary rules means the best explanation splits or merges words: not tolerated
    if _relaxed_search(ws, t, cum, gap_ws, limit, check_boundaries=False)[1] < edits:
        return failed
    return {"labels": labels, "status": "ok_relaxed", "reason": f"{edits} edits after case and width", "edits": edits}


def teacher_stats(text, spans, teacher_text, labels):
    """Quality statistics in units that do not depend on re-segmenting the teacher output.

    The official label_word.py splits the teacher output into words; with our segmenter that
    gives wrong numbers (original ["天", "气"], teacher re-split as ["天气"] -> variation 1).
    - char_variation_rate: share of teacher characters not available in the original (multiset)
    - char_comp_rate: teacher characters / original characters (whitespace removed)
    - word_keep_rate: kept / content words (see content_words), only for ok samples (None otherwise)
    Alignment failures are reported by align() status, not by an alignment_gap number.
    """
    orig = "".join(_strip(text[s:e]) for s, e in spans)
    t, _ = _teacher_chars(teacher_text)
    extra = Counter(t) - Counter(orig)
    idx = content_words(spans)
    return {
        "char_variation_rate": sum(extra.values()) / len(t) if t else 0.0,
        "char_comp_rate": len(t) / len(orig) if orig else 0.0,
        "word_keep_rate": sum(labels[i] for i in idx) / len(idx) if labels and idx else None,
    }


def content_words(spans):
    """Indices of words that contain characters. Whitespace-only words (empty span) never count
    toward n and are never kept (DECISIONS.md B2); budgets and rates must use this."""
    return [i for i, (s, e) in enumerate(spans) if e > s]
