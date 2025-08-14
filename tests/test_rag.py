import unittest
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.vector_store import VectorStore
from src.document_processor import DocumentProcessor
from src.rag_pipeline import RAGPipeline

class TestRAGPipeline(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_collection = "test_collection"
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_vector_store_creation(self):
        vector_store = VectorStore(
            collection_name=self.test_collection,
            persist_directory=self.temp_dir
        )
        
        self.assertEqual(vector_store.collection_name, self.test_collection)
        self.assertEqual(vector_store.persist_directory, self.temp_dir)
        
        info = vector_store.get_collection_info()
        self.assertEqual(info["name"], self.test_collection)
        self.assertEqual(info["count"], 0)
    
    @patch('src.document_processor.OpenAIEmbeddings')
    def test_document_processor_initialization(self, mock_embeddings):
        mock_embeddings.return_value = MagicMock()
        
        processor = DocumentProcessor(
            chunk_size=500,
            chunk_overlap=100,
            embedding_model="text-embedding-3-small"
        )
        
        self.assertEqual(processor.chunk_size, 500)
        self.assertEqual(processor.chunk_overlap, 100)
        mock_embeddings.assert_called_once_with(model="text-embedding-3-small")
    
    def test_text_splitting(self):
        from langchain.schema import Document
        
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        
        long_text = "This is a test document. " * 20
        documents = [Document(page_content=long_text, metadata={"source": "test.txt"})]
        
        split_docs = processor.split_documents(documents)
        
        self.assertGreater(len(split_docs), 1)
        for doc in split_docs:
            self.assertLessEqual(len(doc.page_content), 100 + 50)  # Allow some margin
    
    def create_test_text_file(self, content: str) -> str:
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(content)
        return test_file
    
    @patch('src.document_processor.OpenAIEmbeddings')
    def test_file_processing(self, mock_embeddings):
        mock_embedding_instance = MagicMock()
        mock_embedding_instance.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_embeddings.return_value = mock_embedding_instance
        
        processor = DocumentProcessor()
        test_content = "이것은 테스트 문서입니다. " * 50
        test_file = self.create_test_text_file(test_content)
        
        result = processor.process_file(test_file)
        
        self.assertIn("texts", result)
        self.assertIn("embeddings", result)
        self.assertIn("metadatas", result)
        self.assertIn("ids", result)
        
        self.assertGreater(len(result["texts"]), 0)
        self.assertEqual(len(result["texts"]), len(result["embeddings"]))
        self.assertEqual(len(result["texts"]), len(result["metadatas"]))
        self.assertEqual(len(result["texts"]), len(result["ids"]))

if __name__ == '__main__':
    unittest.main()