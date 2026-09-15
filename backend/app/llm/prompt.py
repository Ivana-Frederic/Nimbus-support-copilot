from app.rag.retriever import RetrievedChunk

SYSTEM_PROMPT = """You are Nimbus Cloud's support copilot. Answer the customer's \
question using ONLY the context passages provided below. If the answer isn't \
in the context, say you don't have that information and suggest contacting \
human support. Do not make anything up. Be concise and friendly. When you \
use a fact from a passage, you don't need to cite it inline; sources are \
shown separately to the user."""


def build_messages(query: str, chunks: list[RetrievedChunk]) -> list[dict[str, str]]:
    if chunks:
        context = "\n\n".join(
            f"[Passage {i + 1} - {c.source}]\n{c.text}" for i, c in enumerate(chunks)
        )
    else:
        context = "(no relevant passages found in the knowledge base)"

    user_message = f"Context:\n{context}\n\nCustomer question: {query}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
