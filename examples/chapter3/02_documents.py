"""3.1.2 文書の構造化

ノートブック: notebooks/chapter3/01_knowledge.ipynb
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plant_documents import PLANT_DOCUMENTS, create_vectorstore

QUERY = "熊童子"


def demo_document() -> None:
    """Document オブジェクトの作成と表示。"""
    document = PLANT_DOCUMENTS[0]
    print(document)


def demo_similarity_search() -> None:
    """Chroma による類似度検索。"""
    vectorstore = create_vectorstore()

    print("=== similarity_search ===")
    for doc in vectorstore.similarity_search(QUERY):
        print(doc.page_content[:50], "...")

    print("\n=== similarity_search_with_score ===")
    for doc, score in vectorstore.similarity_search_with_score(QUERY):
        print(f"score={score:.4f}: {doc.page_content[:50]}...")


def main() -> None:
    print("=== Document ===")
    demo_document()

    print("\n=== 類似度検索 ===")
    demo_similarity_search()


if __name__ == "__main__":
    main()
