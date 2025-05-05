#!/usr/bin/env python3
"""
Tests for document processing pipeline.
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Import the module to test
from pipeline import (
    parse_arguments,
    find_files,
    parse_metadata,
    generate_yaml_frontmatter,
    convert_txt,
)


class TestPipeline(unittest.TestCase):
    """Test cases for pipeline functions."""
    
    def test_parse_arguments_defaults(self):
        """Test argument parser with default values."""
        with patch('sys.argv', ['pipeline.py']):
            args = parse_arguments()
            self.assertEqual(args.output, "combined_output.md")
            self.assertEqual(args.model, "gpt-3.5-turbo")
            self.assertEqual(args.length, 150)
            self.assertFalse(args.summary)
            self.assertIsNone(args.metadata)
            self.assertIsNone(args.tags)

    def test_parse_arguments_custom(self):
        """Test argument parser with custom values."""
        test_args = [
            'pipeline.py',
            '--metadata', 'author=Test User', 'date=2023-01-01',
            '--tags', 'test', 'documentation',
            '--summary',
            '--model', 'gpt-4',
            '--length', '200',
            '--output', 'test_output.md'
        ]
        with patch('sys.argv', test_args):
            args = parse_arguments()
            self.assertEqual(args.output, "test_output.md")
            self.assertEqual(args.model, "gpt-4")
            self.assertEqual(args.length, 200)
            self.assertTrue(args.summary)
            self.assertEqual(args.metadata, ['author=Test User', 'date=2023-01-01'])
            self.assertEqual(args.tags, ['test', 'documentation'])

    @patch('glob.glob')
    def test_find_files(self, mock_glob):
        """Test file discovery."""
        # Mock glob.glob to return test files
        def mock_glob_side_effect(pattern):
            extensions = {
                "*.pdf": ["doc1.pdf", "doc2.pdf"],
                "*.docx": ["report.docx"],
                "*.txt": ["notes.txt"],
                "*.html": ["page.html"],
                "*.htm": [],
                "*.csv": ["data.csv"]
            }
            return extensions.get(pattern, [])
        
        mock_glob.side_effect = mock_glob_side_effect
        
        result = find_files()
        
        self.assertEqual(len(result["pdf"]), 2)
        self.assertEqual(len(result["docx"]), 1)
        self.assertEqual(len(result["txt"]), 1)
        self.assertEqual(len(result["html"]), 1)
        self.assertEqual(len(result["csv"]), 1)

    def test_parse_metadata(self):
        """Test metadata parsing."""
        test_metadata = ["author=Test User", "date=2023-01-01"]
        result = parse_metadata(test_metadata)
        
        self.assertEqual(result["author"], "Test User")
        self.assertEqual(result["date"], "2023-01-01")
        self.assertIn("processing_date", result)
        self.assertIn("processor", result)

    def test_generate_yaml_frontmatter(self):
        """Test YAML frontmatter generation."""
        metadata = {
            "title": "Test Document",
            "author": "Test User",
            "date": "2023-01-01"
        }
        tags = ["test", "documentation"]
        
        result = generate_yaml_frontmatter(metadata, tags)
        
        self.assertIn("---", result)
        self.assertIn("title: Test Document", result)
        self.assertIn("author: Test User", result)
        self.assertIn("date: 2023-01-01", result)
        self.assertIn("tags:", result)
        self.assertIn("  - test", result)
        self.assertIn("  - documentation", result)

    def test_convert_txt(self):
        """Test text file conversion."""
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp:
            temp.write("This is a test content.\nLine 2\nLine 3")
            temp_path = temp.name
        
        try:
            result = convert_txt(temp_path)
            self.assertIn("This is a test content.", result)
            self.assertIn("Line 2", result)
            self.assertIn("Line 3", result)
        finally:
            # Clean up the temporary file
            os.unlink(temp_path)


if __name__ == "__main__":
    unittest.main()