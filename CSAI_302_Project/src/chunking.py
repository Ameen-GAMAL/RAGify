def chunk_text(
    text,
    lecture_id,
    chunk_size=400,
    overlap=80
):
    """
    Split text into overlapping chunks.

    Args:
        text (str): Lecture text
        lecture_id (str): Parent lecture ID
        chunk_size (int): Target number of words per chunk
        overlap (int): Number of overlapping words

    Returns:
        list: List of chunk dictionaries
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []

    current_chunk = []
    current_word_count = 0
    chunk_index = 1

    for paragraph in paragraphs:
        words = paragraph.split()
        paragraph_word_count = len(words)

        if current_word_count + paragraph_word_count <= chunk_size:
            current_chunk.append(paragraph)
            current_word_count += paragraph_word_count
        else:
            chunk_text_content = " ".join(current_chunk)

            chunks.append({
                "chunk_id": f"{lecture_id}_chunk_{chunk_index:02d}",
                "lecture_id": lecture_id,
                "text": chunk_text_content
            })

            chunk_index += 1

            overlap_words = chunk_text_content.split()[-overlap:]
            current_chunk = [" ".join(overlap_words), paragraph]
            current_word_count = len(overlap_words) + paragraph_word_count

    if current_chunk:
        chunks.append({
            "chunk_id": f"{lecture_id}_chunk_{chunk_index:02d}",
            "lecture_id": lecture_id,
            "text": " ".join(current_chunk)
        })

    return chunks
