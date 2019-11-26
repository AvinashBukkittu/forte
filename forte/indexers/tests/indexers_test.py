"""This module tests indexer module."""
import unittest
import shutil
import os
import numpy as np

from forte.indexers import EmbeddingBasedIndexer, ElasticSearchIndexer


class TestEmbeddingBasedIndexer(unittest.TestCase):
    r"""Tests Embedding based indexers."""

    @classmethod
    def setUpClass(cls):
        cls.index = EmbeddingBasedIndexer(
            hparams={"index_type": "IndexFlatL2", "dim": 2})

        vectors = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
        cls.index.add(vectors, meta_data={0: "0", 1: "1", 2: "2"})
        cls.index_path = "index/"

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.index_path):
            shutil.rmtree(cls.index_path)

    def test_indexer(self):
        self.assertEqual(self.index._index.ntotal, 3)
        actual_results = [[np.array([1, 0], dtype=np.float32),
                           np.array([1, 1], dtype=np.float32)],
                          [np.array([1, 0], dtype=np.float32),
                           np.array([0, 1], dtype=np.float32)]]
        search_vectors = np.array([[1, 0], [0.25, 0]], dtype=np.float32)
        results = self.index.search(search_vectors, k=2)
        for i, result in enumerate(results):
            for j, t in enumerate(result):
                self.assertTrue(
                    np.all(actual_results[i][j] ==
                           self.index._index.reconstruct(int(t[0]))))

    def test_indexer_save_and_load(self):
        self.index.save(path=self.index_path)
        saved_files = ["index.faiss", "index.meta_data"]
        for file in os.listdir(path=self.index_path):
            self.assertIn(file, saved_files)

        new_index = EmbeddingBasedIndexer(
            hparams={"index_type": "IndexFlatL2", "dim": 2})
        new_index.load(path=self.index_path, device="cpu")

        self.assertEqual(new_index._index.ntotal, 3)
        actual_results = [[np.array([1, 0], dtype=np.float32),
                           np.array([1, 1], dtype=np.float32)],
                          [np.array([1, 0], dtype=np.float32),
                           np.array([0, 1], dtype=np.float32)]]
        search_vectors = np.array([[1, 0], [0.25, 0]], dtype=np.float32)
        results = new_index.search(search_vectors, k=2)
        for i, result in enumerate(results):
            for j, t in enumerate(result):
                self.assertTrue(
                    np.all(actual_results[i][j] ==
                           new_index._index.reconstruct(int(t[0]))))


class TestElasticSearchIndexer(unittest.TestCase):
    r"""Tests Elastic Indexer."""

    def setUp(self):
        self.indexer = ElasticSearchIndexer()

    def tearDown(self):
        self.indexer.elasticsearch.indices.delete(
            index=self.indexer.hparams.index_name, ignore=[400, 404])

    def test_add(self):
        document = {"key": "This document is created to test "
                           "ElasticSearchIndexer"}
        self.indexer.add(document, refresh="wait_for")
        retrieved_document = self.indexer.search(
            query={"query": {"match": {"key": "ElasticSearchIndexer"}},
                   "_source": ["key"]})
        hits = retrieved_document["hits"]["hits"]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["_source"], document)

    def test_add_bulk(self):
        document = {"key": "This document is created to test "
                           "ElasticSearchIndexer"}
        self.indexer.add_bulk([document], refresh="wait_for")
        retrieved_document = self.indexer.search(
            query={"query": {"match_all": {}}})
        hits = retrieved_document["hits"]["hits"]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["_source"], document)