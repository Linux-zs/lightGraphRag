"""Source-preserving hard splits for text and embedding input limits."""

from typing import Callable

from lightrag.chunker import chunking_by_recursive_character


def embedding_aligned_chunks(tokenizer, content: str, options: dict, *, max_chars: int,
                             max_tokens: int, embedding_length: Callable[[str], int]) -> list[dict]:
    if max_chars <= 0:
        raise ValueError("Embedding character limit must be positive")
    rows = chunking_by_recursive_character(
        tokenizer, content, options['chunk_token_size'], **options['recursive_character'])
    result = []
    for row in rows:
        text = row['content']
        offset = 0
        while offset < len(text):
            end = min(len(text), offset + max_chars)

            def fits(candidate_end):
                candidate = text[offset:candidate_end].strip()
                return max_tokens <= 0 or (
                    len(tokenizer.encode(candidate)) <= max_tokens
                    and embedding_length(candidate) <= max_tokens)

            if not fits(end):
                # Search only character boundaries: token boundaries may split
                # a UTF-8 character. Every accepted prefix is measured directly.
                low, high, best = offset + 1, end - 1, offset
                while low <= high:
                    middle = (low + high) // 2
                    if fits(middle):
                        best, low = middle, middle + 1
                    else:
                        high = middle - 1
                end = best
            if end == offset:
                raise ValueError("Embedding token limit cannot fit a single source character")
            raw_piece = text[offset:end]
            piece = raw_piece.strip()
            if piece:
                item = {**row, 'content': piece, 'tokens': len(tokenizer.encode(piece)),
                        'chunk_order_index': len(result)}
                parent = row.get('_source_span')
                if isinstance(parent, dict) and content[parent['start']:parent['end']] == text:
                    start = parent['start'] + offset + len(raw_piece) - len(raw_piece.lstrip())
                    item['_source_span'] = {'start': start, 'end': start + len(piece)}
                else:
                    item.pop('_source_span', None)
                result.append(item)
            offset = end
    return result
