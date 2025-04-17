import argparse
import asyncio

from src.vector_db_manager.chroma_manager import ChromaManager
from src.vector_db_manager.image_embedder import ImageEmbedder
from src.vector_db_manager.text_embedder import TextEmbedder

"""
# Chroma CLI Inspector

A command line tool for inspecting and managing ChromaDB collections.

## Overview

This utility provides a simple interface to:
- List all collections in a ChromaDB instance
- Inspect the contents of a specific collection
- View sample documents across all collections

The tool leverages the `ChromaManager` class which provides a wrapper around ChromaDB operations, along with text and image embedding capabilities.

## Prerequisites

- Python 3.7+
- ChromaDB
- Required dependencies (installed via requirements.txt or setup.py)

## Installation

1. Clone the repository
2. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

```bash
# List all collections in the database
python chroma_inspector.py --list

# Inspect a specific collection (shows 10 items by default)
python chroma_inspector.py --collection my_collection_name

# Inspect a specific collection with a custom limit
python chroma_inspector.py --collection my_collection_name --limit 20

# Inspect all collections (shows up to 5 items per collection)
python chroma_inspector.py --all
```

## Arguments

- `--list`: Lists all available collections with their item counts
- `--collection NAME`: Specifies a collection to inspect
- `--limit N`: Limits the number of items shown (default: 10)
- `--all`: Inspects all collections and prints up to 5 documents each

## Functions

### `list_collections(manager)`
Prints all available collections and their item counts.

### `inspect_collection(manager, collection_name, limit=10)`
Displays detailed information about items in a specific collection.

### `inspect_all_data(manager)`
Shows sample documents (up to 5) from each collection in the database.

### `main(args)`
The main entry point that processes command line arguments and calls the appropriate functions.

## Example Output

```
[INFO] Using Chroma persist directory: ./chroma_db

Available collections:
- product_descriptions: 124 items
- customer_reviews: 532 items
- image_catalog: 78 items

Inspecting collection: product_descriptions
ID: 1a2b3c4d-5e6f-7g8h-9i0j-1k2l3m4n5o6p
Metadata: {'source': 'website', 'category': 'electronics', 'date_added': '2023-05-15'}
Document: High-performance laptop with 16GB RAM and 512GB SSD storage...
```

## Configuration

The tool uses a default ChromaDB persist directory of `./chroma_db`. If you need to use a different directory, you can modify the `persist_directory` parameter in the code.

## Dependencies

- `src.vector_db_manager.chroma_manager`: Provides the ChromaManager class
- `src.vector_db_manager.image_embedder`: Handles image embedding functionality
- `src.vector_db_manager.text_embedder`: Handles text embedding functionality

## Note

This tool is designed for inspection purposes only. It does not provide functionality for modifying or deleting data in the ChromaDB instance.
"""

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
