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
    print(f"\nInspecting collection: {collection_name}")
    result = await manager.get(collection_name=collection_name, limit=limit, include=["metadatas", "documents"])

    if not result["results"]:
        print("No documents found in this collection.")
    else:
        for item in result["results"]:
            print(f"\nID: {item.get('id')}")
            print(f"Metadata: {item.get('metadata')}")
            print(f"Document: {item.get('document')}")


async def inspect_all_data(manager: ChromaManager):
    print("Inspecting all collections and printing sample documents...\n")
    for name in manager.collections:
        print(f"\n>>> Collection: {name}")
        result = await manager.get(collection_name=name, limit=5, include=["documents", "metadatas"])

        if not result["results"]:
            print("No data found in this collection.")
        else:
            for item in result["results"]:
                print(f"\nID: {item.get('id')}")
                print(f"Document: {item.get('document')}")
                print(f"Metadata: {item.get('metadata')}")


async def main(args):
    text_embedder = TextEmbedder()
    image_embedder = ImageEmbedder()
    manager = ChromaManager(
        text_embedder=text_embedder,
        image_embedder=image_embedder,
        persist_directory="./chroma_db"
    )

    print(f"[INFO] Using Chroma persist directory: {manager.persist_directory}")

    if args.list:
        await list_collections(manager)
    elif args.collection:
        await inspect_collection(manager, args.collection, args.limit or 10)
    elif args.all:
        await inspect_all_data(manager)
    else:
        print("Please specify --list, --collection, or --all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chroma CLI Inspector")
    parser.add_argument("--list", action="store_true", help="List all collections")
    parser.add_argument("--collection", type=str, help="Specify the collection name to inspect")
    parser.add_argument("--limit", type=int, help="Limit the number of items shown (default 10)")
    parser.add_argument("--all", action="store_true", help="Inspect all collections and print up to 5 documents each")

    args = parser.parse_args()
    asyncio.run(main(args))
