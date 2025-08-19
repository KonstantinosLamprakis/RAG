from unittest.mock import Mock, patch

from src.vector_store import setup_persistent_chroma, update_vector_database


class TestVectorStore:

    @patch("src.vector_store.chromadb.PersistentClient")
    def test_setup_persistent_chroma_existing_collection(
        self, mock_client_class, mock_embedding_model
    ):
        """Test setting up ChromaDB with existing collection"""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client

        result = setup_persistent_chroma(mock_embedding_model)

        assert result == mock_collection
        mock_client.get_collection.assert_called_once()

    @patch("src.vector_store.chromadb.PersistentClient")
    def test_setup_persistent_chroma_new_collection(
        self, mock_client_class, mock_embedding_model
    ):
        """Test setting up ChromaDB with new collection"""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_client.create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client

        result = setup_persistent_chroma(mock_embedding_model)

        assert result == mock_collection
        mock_client.create_collection.assert_called_once()

    def test_update_vector_database_no_changes(self, mock_chroma_collection):
        """Test updating vector database with no file changes"""
        mock_doc_manager = Mock()
        mock_doc_manager.get_changed_files.return_value = []

        result = update_vector_database(mock_chroma_collection, mock_doc_manager)

        assert result == 0
        mock_chroma_collection.add.assert_not_called()

    @patch("src.vector_store.process_file_for_rag")
    @patch("src.vector_store.generate_document_id")
    def test_update_vector_database_new_file(
        self, mock_gen_id, mock_process_file, mock_chroma_collection
    ):
        """Test updating vector database with new file"""
        mock_doc_manager = Mock()
        mock_doc_manager.get_changed_files.return_value = [
            {
                "action": "update",
                "file_path": "/test/new_file.txt",
                "file_info": {"filename": "new_file.txt", "hash": "abc123"},
            }
        ]

        mock_process_file.return_value = [
            {"content": "Test content", "metadata": {"source_file": "new_file.txt"}}
        ]

        mock_gen_id.return_value = "test_id_1"
        mock_chroma_collection.get.return_value = {"ids": []}

        result = update_vector_database(mock_chroma_collection, mock_doc_manager)

        assert result == 1
        mock_chroma_collection.add.assert_called_once()
        mock_doc_manager.update_file_metadata.assert_called_once()

    def test_update_vector_database_deleted_file(self, mock_chroma_collection):
        """Test updating vector database with deleted file"""
        mock_doc_manager = Mock()
        mock_doc_manager.get_changed_files.return_value = [
            {
                "action": "delete",
                "file_path": "/test/deleted_file.txt",
                "file_info": None,
            }
        ]

        mock_chroma_collection.get.return_value = {"ids": ["id1", "id2"]}

        result = update_vector_database(mock_chroma_collection, mock_doc_manager)

        assert result == 1
        mock_chroma_collection.delete.assert_called_once_with(ids=["id1", "id2"])
        mock_doc_manager.remove_file_metadata.assert_called_once()
