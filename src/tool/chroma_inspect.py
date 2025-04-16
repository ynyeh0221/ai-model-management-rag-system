# inspect_chroma.py

import argparse
import asyncio

from src.vector_db_manager.chroma_manager import ChromaManager
from src.vector_db_manager.image_embedder import ImageEmbedder
from src.vector_db_manager.text_embedder import TextEmbedder


async def list_collections(manager: ChromaManager):
    print("Available collections:")
    for name in manager.collections.keys():
        stats = await manager.get_collection_stats(name)
        print(f"- {stats['name']}: {stats['count']} items")


async def inspect_collection(manager: ChromaManager, collection_name: str, limit: int = 10):
    print(f"Inspecting collection: {collection_name}")
    result = await manager.get(collection_name=collection_name, limit=limit, include=["metadatas", "documents"])
    for item in result["results"]:
        print(f"\nID: {item.get('id')}")
        print(f"Metadata: {item.get('metadata')}")
        print(f"Document: {item.get('document')}")


async def main(args):
    # Initialize embedders nad  ChromaManager
    text_embedder = TextEmbedder()
    image_embedder = ImageEmbedder()
    manager = ChromaManager(text_embedder=text_embedder, image_embedder=image_embedder)

    if args.list:
        await list_collections(manager)
    elif args.collection:
        await inspect_collection(manager, args.collection, args.limit or 10)
    else:
        print("Please choose --list or --collection")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chroma CLI Inspector")
    parser.add_argument("--list", action="store_true", help="List all collections")
    parser.add_argument("--collection", type=str, help="Specify the collection name to be checked")
    parser.add_argument("--limit", type=int, help="Limit on the output count")

    args = parser.parse_args()
    asyncio.run(main(args))
