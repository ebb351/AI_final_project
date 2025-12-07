#!/usr/bin/env python3
"""
MongoDB Setup and Initialization for GraphRAG

Initializes the MongoDB database with proper schemas, indexes, and validation.
"""
import logging
import os
from datetime import datetime
from pymongo import MongoClient, IndexModel
from pymongo.errors import CollectionInvalid, OperationFailure
from dotenv import load_dotenv

from .schema import get_collection_schemas, get_collection_indexes

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MongoDB configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "graphrag_course_db")


class MongoDBSetup:
    """Handles MongoDB database initialization."""

    def __init__(self, uri: str = MONGODB_URI, database: str = MONGODB_DATABASE):
        """
        Initialize MongoDB setup.

        Args:
            uri: MongoDB connection URI
            database: Database name
        """
        self.uri = uri
        self.database_name = database
        self.client = None
        self.db = None

    def connect(self):
        """Connect to MongoDB."""
        logger.info(f"Connecting to MongoDB at {self.uri}...")
        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.database_name]
            logger.info(f"Connected to database: {self.database_name}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    def create_collections(self):
        """Create collections with validation schemas."""
        logger.info("Creating collections with validation...")

        schemas = get_collection_schemas()

        for collection_name, schema in schemas.items():
            try:
                # Try to create collection with validation
                self.db.create_collection(collection_name, **schema)
                logger.info(f"Created collection: {collection_name}")
            except CollectionInvalid:
                # Collection already exists
                logger.info(f"Collection already exists: {collection_name}")

                # Update validation schema
                try:
                    self.db.command({
                        "collMod": collection_name,
                        **schema
                    })
                    logger.info(f"Updated validation for: {collection_name}")
                except Exception as e:
                    logger.warning(f"Could not update validation for {collection_name}: {e}")

            except Exception as e:
                logger.error(f"Error creating collection {collection_name}: {e}")

    def create_indexes(self):
        """Create indexes for all collections."""
        logger.info("Creating indexes...")

        indexes = get_collection_indexes()

        for collection_name, index_definitions in indexes.items():
            collection = self.db[collection_name]

            for index_def in index_definitions:
                try:
                    index_model = IndexModel(**index_def)
                    collection.create_indexes([index_model])
                    logger.info(f"Created index {index_def['name']} on {collection_name}")
                except Exception as e:
                    logger.warning(f"Index {index_def['name']} on {collection_name} may already exist: {e}")

    def initialize(self):
        """Initialize database with schemas and indexes."""
        logger.info("=" * 80)
        logger.info("MONGODB INITIALIZATION")
        logger.info("=" * 80)

        # Connect
        self.connect()

        # Create collections
        self.create_collections()

        # Create indexes
        self.create_indexes()

        # Show database stats
        self.show_stats()

        logger.info("=" * 80)
        logger.info("MongoDB initialization complete!")
        logger.info("=" * 80)

    def show_stats(self):
        """Show database statistics."""
        logger.info("\nDatabase Statistics:")
        logger.info(f"  Database: {self.database_name}")

        collections = self.db.list_collection_names()
        logger.info(f"  Collections: {len(collections)}")

        for collection_name in sorted(collections):
            count = self.db[collection_name].count_documents({})
            logger.info(f"    - {collection_name}: {count} documents")

    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("Closed MongoDB connection")


def main():
    """Main function to initialize MongoDB."""
    setup = MongoDBSetup()

    try:
        setup.initialize()
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return 1
    finally:
        setup.close()

    return 0


if __name__ == "__main__":
    exit(main())
