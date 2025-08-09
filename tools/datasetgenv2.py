#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced Dataset Generator for LLM Fine-Tuning - Version 2 (Offline)
-------------------------------------------------------------------
A comprehensive GUI toolkit for generating, formatting, and processing
datasets specifically designed for fine-tuning language models.
Features include format conversion, auto-fixing, data augmentation,
visualization, and support for numerous file formats.

This version is designed for completely offline use, with NO external API
access, NO embedding models, and NO third-party connections. All
functionality is contained within this single file.

Version 2 Enhancements:
- Enhanced GUI with modern design and improved interactivity.
- Interactive field mapping dialog with smart suggestions.
- Local data generation (using faker and random data).
- Enhanced local data augmentation (rephrasing, synonym replacement, shuffling).
- Direct table editing in data views.
- Search and filtering capabilities for data tables.
- Custom validation rules (regex, type, range, required fields).
- More comprehensive statistics and visualization options.
- Improved error handling and user feedback.
"""

import os
import sys
import json
import csv
import argparse
import re
import random
import sqlite3
import tempfile
import datetime
import uuid
import logging
import threading
import math
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
import pandas as pd
import numpy as np
from tqdm import tqdm
# Rich is primarily for CLI, but keeping imports for consistency if CLI mode is used
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich import print as rprint

# For GUI
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QLabel, QPushButton, QComboBox, QLineEdit, QTextEdit,
                           QTabWidget, QFileDialog, QMessageBox, QProgressBar,
                           QTableWidget, QTableWidgetItem, QSplitter, QCheckBox,
                           QGroupBox, QRadioButton, QScrollArea, QSlider, QSpinBox,
                           QListWidget, QListWidgetItem, QTreeWidget, QTreeWidgetItem,
                           QDockWidget, QStackedWidget, QToolBar, QStatusBar, QDialog,
                           QWizard, QWizardPage, QFormLayout, QGridLayout, QMenu, QAction,
                           QAbstractTableModel, QToolTip, QHeaderView, QInputDialog, QCompleter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl, QSize, QTimer, QSettings, QMimeData, QStringListModel
from PyQt5.QtGui import QIcon, QFont, QDrag, QPixmap, QColor, QPainter, QPalette, QDesktopServices

# For advanced file formats (optional, checked at runtime)
try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    print("Warning: 'pyarrow' not found. Parquet format support disabled. Install with 'pip install pyarrow'.")

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    print("Warning: 'h5py' not found. HDF5 format support disabled. Install with 'pip install h5py'.")

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("Warning: 'PyYAML' not found. YAML format support disabled. Install with 'pip install PyYAML'.")

try:
    import xml.etree.ElementTree as ET
    from lxml import etree # lxml for better XML parsing/serialization
    XML_AVAILABLE = True
except ImportError:
    XML_AVAILABLE = False
    print("Warning: 'lxml' not found. Enhanced XML format support disabled. Install with 'pip install lxml'.")
    # Fallback to basic ET if lxml is not available but ET is
    try:
        import xml.etree.ElementTree as ET
        XML_AVAILABLE_BASIC = True
    except ImportError:
        XML_AVAILABLE_BASIC = False

try:
    import fastavro
    AVRO_AVAILABLE = True
except ImportError:
    AVRO_AVAILABLE = False
    print("Warning: 'fastavro' not found. Avro format support disabled. Install with 'pip install fastavro'.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: 'matplotlib' and 'seaborn' not found. Visualization features disabled. Install with 'pip install matplotlib seaborn'.")

try:
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: 'scikit-learn' not found. Dataset splitting features disabled. Install with 'pip install scikit-learn'.")

try:
    from faker import Faker
    FAKER_AVAILABLE = True
    faker = Faker()
except ImportError:
    FAKER_AVAILABLE = False
    print("Warning: 'Faker' not found. Auto-generation and anonymization features disabled. Install with 'pip install Faker'.")

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import wordnet
    # Download NLTK data if not already present
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except nltk.downloader.DownloadError:
        nltk.download('wordnet', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: 'nltk' not found. Advanced text augmentation features disabled. Install with 'pip install nltk' and download 'punkt' and 'wordnet' corpora.")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dataset_generator_v2.log')
    ]
)
logger = logging.getLogger("DatasetGeneratorV2")

# Initialize Rich console (for CLI mode fallback)
console = Console()

# Define supported output formats
OUTPUT_FORMATS = {
    "jsonl": "JSON Lines format (each line is a valid JSON object)",
    "json": "Single JSON file with array of examples",
    "csv": "CSV format with headers",
    "txt": "Plain text format",
    "parquet": "Apache Parquet columnar storage format",
    "hdf5": "Hierarchical Data Format version 5",
    "sqlite": "SQLite database file",
    "xml": "XML format",
    "yaml": "YAML format",
    "avro": "Apache Avro data serialization format",
    "excel": "Microsoft Excel format (.xlsx)",
    "tsv": "Tab-separated values format",
}

# Define supported input formats
INPUT_FORMATS = {
    "json": "JSON file (array or object)",
    "jsonl": "JSON Lines file (each line is a JSON object)",
    "csv": "CSV file with headers",
    "txt": "Plain text file",
    "excel": "Excel file (.xlsx, .xls)",
    "parquet": "Apache Parquet columnar storage format",
    "hdf5": "Hierarchical Data Format version 5",
    "sqlite": "SQLite database file (requires table name)",
    "xml": "XML format",
    "yaml": "YAML format",
    "avro": "Apache Avro data serialization format",
    "tsv": "Tab-separated values format",
}

# Define common fine-tuning dataset templates
DATASET_TEMPLATES = {
    "instruction": {
        "description": "Simple instruction-following format (prompt, completion)",
        "structure": {"prompt": "", "completion": ""}
    },
    "chat": {
        "description": "Chat conversation format (messages: user, assistant)",
        "structure": {"messages": [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}]}
    },
    "qa": {
        "description": "Question-Answer format (question, answer)",
        "structure": {"question": "", "answer": ""}
    },
    "openai_fine_tuning": {
        "description": "OpenAI fine-tuning format (system, user, assistant messages)",
        "structure": {"messages": [{"role": "system", "content": ""}, {"role": "user", "content": ""}, {"role": "assistant", "content": ""}]}
    },
    "anthropic_format": {
        "description": "Anthropic Claude format (input, output)",
        "structure": {"input": "", "output": ""}
    },
    "alpaca": {
        "description": "Alpaca instruction format (instruction, input, output)",
        "structure": {"instruction": "", "input": "", "output": ""}
    },
    "llama": {
        "description": "Llama instruction format (single 'text' field)",
        "structure": {"text": ""}
    },
    "mistral": {
        "description": "Mistral instruction format (instruction, response)",
        "structure": {"instruction": "", "response": ""}
    },
    "custom": {
        "description": "Custom format defined by user",
        "structure": {}
    }
}

# --- Utility Functions ---
def get_nested_value(data: Dict, path: str, default=None):
    """Retrieves a nested value from a dictionary using a dot-notation path."""
    parts = path.split('.')
    current = data
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            try:
                current = current[int(part)]
            except IndexError:
                return default # Index out of bounds for list
        else:
            return default
    return current

def set_nested_value(data: Dict, path: str, value: Any):
    """Sets a nested value in a dictionary using a dot-notation path.
    Creates intermediate dictionaries/lists if they don't exist.
    """
    parts = path.split('.')
    current = data
    for i, part in enumerate(parts):
        if i == len(parts) - 1:
            current[part] = value
        else:
            if part not in current or not isinstance(current[part], (dict, list)):
                # Try to infer if the next part is an array index
                if i + 1 < len(parts) and parts[i+1].isdigit():
                    current[part] = []
                else:
                    current[part] = {}
            current = current[part]

class DatasetGenerator:
    """Core logic for dataset generation and processing."""

    def __init__(self):
        self.input_data = []
        self.output_data = []
        self.input_format = None
        self.output_format = None
        self.template = None
        self.mapping = {}
        self.errors = []
        self.warnings = []
        self.statistics = {}

    def _reset_state(self):
        """Resets the internal data and state variables."""
        self.input_data = []
        self.output_data = []
        self.input_format = None
        self.output_format = None
        self.template = None
        self.mapping = {}
        self.errors = []
        self.warnings = []
        self.statistics = {}

    def load_data(self, file_path: str, format_type: Optional[str] = None, table_name: Optional[str] = None) -> None:
        """Load data from input file."""
        self._reset_state() # Reset state before loading new data

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")

        # Determine format based on file extension if not specified
        if not format_type:
            ext = os.path.splitext(file_path)[1].lower()[1:]
            if ext in ['xlsx', 'xls']:
                format_type = 'excel'
            elif ext in INPUT_FORMATS:
                format_type = ext
            else:
                raise ValueError(f"Unsupported file format: {ext}")

        self.input_format = format_type
        logger.info(f"Loading data from {file_path} as {format_type} format")

        try:
            if format_type == 'json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        if 'data' in data and isinstance(data['data'], list):
                            self.input_data = data['data']
                        else:
                            self.input_data = [data] # Wrap single object in a list
                    else:
                        self.input_data = data

            elif format_type == 'jsonl':
                self.input_data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            self.input_data.append(json.loads(line))

            elif format_type == 'csv':
                df = pd.read_csv(file_path)
                self.input_data = df.to_dict(orient='records')

            elif format_type == 'tsv':
                df = pd.read_csv(file_path, sep='\t')
                self.input_data = df.to_dict(orient='records')

            elif format_type == 'excel':
                df = pd.read_excel(file_path)
                self.input_data = df.to_dict(orient='records')

            elif format_type == 'txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    self.input_data = [{"text": line.strip()} for line in lines if line.strip()]

            elif format_type == 'parquet':
                if not PARQUET_AVAILABLE:
                    raise ImportError("PyArrow library not available for Parquet format.")
                table = pq.read_table(file_path)
                df = table.to_pandas()
                self.input_data = df.to_dict(orient='records')

            elif format_type == 'hdf5':
                if not HDF5_AVAILABLE:
                    raise ImportError("h5py library not available for HDF5 format.")
                with h5py.File(file_path, 'r') as f:
                    dataset_name = list(f.keys())[0] if not table_name else table_name
                    obj = f[dataset_name]
                    if isinstance(obj, h5py.Dataset):
                        data = obj[:]
                        if hasattr(obj, 'dtype') and obj.dtype.names:
                            self.input_data = [{name: item[name] for name in obj.dtype.names} for item in data]
                        else:
                            self.input_data = [{"value": item} for item in data]
                    else:
                        # If it's a group, try to find the first dataset inside the group
                        found_dataset = None
                        for key in obj.keys():
                            if isinstance(obj[key], h5py.Dataset):
                                found_dataset = obj[key]
                                break
                        if found_dataset is not None:
                            data = found_dataset[:]
                            if hasattr(found_dataset, 'dtype') and found_dataset.dtype.names:
                                self.input_data = [{name: item[name] for name in found_dataset.dtype.names} for item in data]
                            else:
                                self.input_data = [{"value": item} for item in data]
                        else:
                            raise ValueError(f"No dataset found in HDF5 group '{dataset_name}'")

            elif format_type == 'sqlite':
                conn = sqlite3.connect(file_path)
                conn.row_factory = sqlite3.Row

                if not table_name:
                    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    if not tables:
                        raise ValueError("No tables found in SQLite database")
                    table_name = tables[0][0]

                cursor = conn.execute(f"SELECT * FROM {table_name}")
                self.input_data = [dict(row) for row in cursor.fetchall()]
                conn.close()

            elif format_type == 'xml':
                if not XML_AVAILABLE:
                    raise ImportError("lxml library not available for XML format.")
                tree = ET.parse(file_path)
                root = tree.getroot()
                children = list(root)
                if not children:
                    self.input_data = [self._xml_to_dict(root)]
                else:
                    self.input_data = [self._xml_to_dict(child) for child in children]

            elif format_type == 'yaml':
                if not YAML_AVAILABLE:
                    raise ImportError("PyYAML library not available for YAML format.")
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, list):
                        self.input_data = data
                    elif isinstance(data, dict):
                        if 'data' in data:
                            self.input_data = data['data']
                        else:
                            self.input_data = [data]

            elif format_type == 'avro':
                if not AVRO_AVAILABLE:
                    raise ImportError("fastavro library not available for Avro format.")
                records = []
                with open(file_path, 'rb') as f:
                    for record in fastavro.reader(f):
                        records.append(record)
                self.input_data = records

            else:
                raise ValueError(f"Unsupported input format: {format_type}")

            logger.info(f"Successfully loaded {len(self.input_data)} records")
            self._calculate_statistics()

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            self.errors.append(f"Error loading data: {str(e)}")
            raise

    def _xml_to_dict(self, element):
        """Convert XML element to dictionary."""
        result = {}
        for key, value in element.attrib.items():
            result[f"@{key}"] = value
        if element.text and element.text.strip():
            result["#text"] = element.text.strip()
        for child in element:
            child_data = self._xml_to_dict(child)
            if child.tag in result:
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data
        return result

    def _calculate_statistics(self):
        """Calculate statistics about the dataset."""
        if not self.input_data:
            self.statistics = {}
            return

        self.statistics = {
            "record_count": len(self.input_data),
            "fields": {},
            "field_types": {},
            "missing_values": {},
            "common_values": {}
        }

        all_field_names = set()
        for item in self.input_data:
            all_field_names.update(item.keys())

        for field_str in all_field_names:
            values = [item.get(field_str) for item in self.input_data if field_str in item]
            non_null_values = [v for v in values if v is not None]

            missing_count = len(values) - len(non_null_values)
            self.statistics["missing_values"][field_str] = missing_count

            if non_null_values:
                sample_value = non_null_values[0]
                if isinstance(sample_value, str):
                    self.statistics["field_types"][field_str] = "string"
                elif isinstance(sample_value, (int, float)):
                    self.statistics["field_types"][field_str] = "number"
                elif isinstance(sample_value, bool):
                    self.statistics["field_types"][field_str] = "boolean"
                elif isinstance(sample_value, list):
                    self.statistics["field_types"][field_str] = "array"
                elif isinstance(sample_value, dict):
                    self.statistics["field_types"][field_str] = "object"
                else:
                    self.statistics["field_types"][field_str] = "unknown"

                if isinstance(sample_value, str) and len(non_null_values) > 0:
                    counter = Counter(non_null_values)
                    self.statistics["common_values"][field_str] = counter.most_common(5)

    def set_template(self, template_name: str) -> None:
        """Set the output template structure."""
        if template_name not in DATASET_TEMPLATES:
            raise ValueError(f"Unknown template: {template_name}")
        self.template = template_name
        logger.info(f"Using template: {template_name}")

    def set_field_mapping(self, mapping: Dict[str, str]) -> None:
        """Set the field mapping."""
        self.mapping = mapping
        logger.info(f"Field mapping set: {mapping}")

    def process_data(self, input_data: List[Dict[str, Any]], template_name: str, mapping: Dict[str, str], num_workers: Optional[int] = None) -> List[Dict[str, Any]]:
        """Process input data according to the template and mapping."""
        if not template_name or not mapping:
            raise ValueError("Template and field mapping must be set before processing")

        logger.info("Processing data...")
        template_structure = DATASET_TEMPLATES[template_name]["structure"]
        self.output_data = []

        if not input_data:
            return []

        # Determine number of workers
        if num_workers is None:
            import multiprocessing
            num_workers = max(1, multiprocessing.cpu_count() - 1)

        if len(input_data) < 1000 or num_workers <= 1:
            for item in tqdm(input_data, desc="Processing records"):
                output_item = self._map_item_to_template(item, template_structure, mapping)
                self.output_data.append(output_item)
        else:
            chunk_size = math.ceil(len(input_data) / num_workers)
            data_chunks = [input_data[i:i + chunk_size] for i in range(0, len(input_data), chunk_size)]

            logger.info(f"Parallel processing with {num_workers} workers ({len(data_chunks)} chunks)...")
            from concurrent.futures import ThreadPoolExecutor, as_completed

            processed_chunks = []
            def process_chunk(chunk):
                result = []
                for item in chunk:
                    output_item = self._map_item_to_template(item, template_structure, mapping)
                    result.append(output_item)
                return result

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(process_chunk, chunk): i for i, chunk in enumerate(data_chunks)}
                for future in as_completed(futures):
                    chunk_idx = futures[future]
                    try:
                        processed_chunk = future.result()
                        processed_chunks.append((chunk_idx, processed_chunk))
                    except Exception as e:
                        logger.error(f"Error processing chunk {chunk_idx}: {str(e)}")

            processed_chunks.sort(key=lambda x: x[0])
            for _, chunk in processed_chunks:
                self.output_data.extend(chunk)

        logger.info(f"Successfully processed {len(self.output_data)} records")
        return self.output_data

    def _map_item_to_template(self, item: Dict[str, Any], template_structure: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
        """Map a single input item to the template structure."""
        result = {}

        def _map_structure(structure, result_dict, prefix=""):
            for key, value in structure.items():
                full_key = f"{prefix}{key}"
                if isinstance(value, dict):
                    result_dict[key] = {}
                    _map_structure(value, result_dict[key], f"{full_key}.")
                elif isinstance(value, list) and value and isinstance(value[0], dict):
                    result_dict[key] = []
                    for i, item_template in enumerate(value):
                        # For chat-like structures, we might need to create multiple items
                        # For simplicity, we assume a fixed number of elements as per template
                        # More complex logic needed for dynamic array lengths
                        result_dict[key].append({})
                        _map_structure(item_template, result_dict[key][i], f"{full_key}[{i}].")
                else:
                    if full_key in mapping:
                        input_field = mapping[full_key]
                        # Handle nested input fields if necessary (e.g., "data.title")
                        result_dict[key] = get_nested_value(item, input_field, "")
                    else:
                        result_dict[key] = "" # Default empty string if no mapping

        _map_structure(template_structure, result)
        return result

    def save_data(self, data: List[Dict[str, Any]], output_path: str, format_type: str) -> None:
        """Save processed data to output file."""
        if not data:
            raise ValueError("No data to save")

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)

        logger.info(f"Saving data to {output_path} as {format_type} format")

        try:
            if format_type == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

            elif format_type == 'jsonl':
                with open(output_path, 'w', encoding='utf-8') as f:
                    for item in data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')

            elif format_type == 'csv':
                df = pd.DataFrame(data)
                df.to_csv(output_path, index=False)

            elif format_type == 'tsv':
                df = pd.DataFrame(data)
                df.to_csv(output_path, sep='\t', index=False)

            elif format_type == 'excel':
                df = pd.DataFrame(data)
                df.to_excel(output_path, index=False)

            elif format_type == 'txt':
                with open(output_path, 'w', encoding='utf-8') as f:
                    for item in data:
                        if isinstance(item, dict):
                            f.write(json.dumps(item, ensure_ascii=False) + '\n')
                        else:
                            f.write(str(item) + '\n')

            elif format_type == 'parquet':
                if not PARQUET_AVAILABLE:
                    raise ImportError("PyArrow library not available for Parquet format.")
                df = pd.DataFrame(data)
                table = pa.Table.from_pandas(df)
                pq.write_table(table, output_path)

            elif format_type == 'hdf5':
                if not HDF5_AVAILABLE:
                    raise ImportError("h5py library not available for HDF5 format.")
                df = pd.DataFrame(data)
                with h5py.File(output_path, 'w') as f:
                    # Convert DataFrame to a structured numpy array and save
                    # This conversion can be complex for mixed types; simplifying for common cases
                    dt = np.dtype([(name, 'U256') for name in df.columns]) # Use unicode string for flexibility
                    dataset = f.create_dataset('data', shape=(len(df),), dtype=dt)
                    for i, row in enumerate(df.itertuples(index=False)):
                        for j, column in enumerate(df.columns):
                            dataset[column][i] = str(row[j]) # Convert all to string for 'U256'

            elif format_type == 'sqlite':
                df = pd.DataFrame(data)
                conn = sqlite3.connect(output_path)
                df.to_sql('data', conn, if_exists='replace', index=False)
                conn.close()

            elif format_type == 'xml':
                if not XML_AVAILABLE:
                    raise ImportError("lxml library not available for XML format.")
                root = etree.Element("dataset") # Use lxml for better XML control

                for item in data:
                    record = etree.SubElement(root, "record")
                    def _add_xml_element(parent, data_item):
                        if isinstance(data_item, dict):
                            for k, v in data_item.items():
                                if isinstance(v, (dict, list)):
                                    child = etree.SubElement(parent, k)
                                    _add_xml_element(child, v)
                                else:
                                    child = etree.SubElement(parent, k)
                                    child.text = str(v)
                        elif isinstance(data_item, list):
                            for sub_item in data_item:
                                # For lists, create a generic 'item' tag or infer from context
                                child = etree.SubElement(parent, "item")
                                _add_xml_element(child, sub_item)

                    _add_xml_element(record, item)

                tree = etree.ElementTree(root)
                tree.write(output_path, pretty_print=True, encoding='utf-8', xml_declaration=True)

            elif format_type == 'yaml':
                if not YAML_AVAILABLE:
                    raise ImportError("PyYAML library not available for YAML format.")
                with open(output_path, 'w', encoding='utf-8') as f:
                    yaml.dump(data, f, default_flow_style=False)

            elif format_type == 'avro':
                if not AVRO_AVAILABLE:
                    raise ImportError("fastavro library not available for Avro format.")
                if not data:
                    raise ValueError("No data to save for Avro.")

                # Generate schema from the first record
                schema_dict = {
                    "namespace": "example.avro",
                    "type": "record",
                    "name": "Dataset",
                    "fields": []
                }

                first_record = data[0]
                for key, value in first_record.items():
                    field_type = "string"  # Default type
                    if isinstance(value, int):
                        field_type = "long" # Avro 'int' is 32-bit, 'long' is 64-bit
                    elif isinstance(value, float):
                        field_type = "double"
                    elif isinstance(value, bool):
                        field_type = "boolean"
                    elif isinstance(value, list):
                        # Avro arrays need a specific item type. Simplistic: assume strings
                        field_type = {"type": "array", "items": "string"}
                    elif isinstance(value, dict):
                        # Avro maps need a specific value type. Simplistic: assume strings
                        field_type = {"type": "map", "values": "string"}

                    schema_dict["fields"].append({
                        "name": key,
                        "type": ["null", field_type] if value is None else field_type
                    })

                schema = fastavro.parse_schema(schema_dict)

                with open(output_path, 'wb') as f:
                    fastavro.writer(f, schema, data)

            else:
                raise ValueError(f"Unsupported output format: {format_type}")

            logger.info(f"Successfully saved {len(data)} records to {output_path}")

        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            self.errors.append(f"Error saving data: {str(e)}")
            raise

    def split_dataset(self, data: List[Dict[str, Any]], train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, random_seed=None):
        """Split dataset into train, validation, and test sets."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn library not available for dataset splitting.")

        if not data:
            raise ValueError("No data to split.")

        # Normalize ratios
        total = train_ratio + valid_ratio + test_ratio
        train_ratio /= total
        valid_ratio /= total
        test_ratio /= total

        # Ensure sum of ratios is 1 (or close to it due to float precision)
        if not (0.99 <= (train_ratio + valid_ratio + test_ratio) <= 1.01):
             logger.warning(f"Ratios sum to {train_ratio + valid_ratio + test_ratio}, normalizing.")

        train_data, temp_data = train_test_split(
            data, test_size=(valid_ratio + test_ratio), random_state=random_seed
        )

        if valid_ratio > 0 and test_ratio > 0:
            valid_data, test_data = train_test_split(
                temp_data, test_size=(test_ratio / (valid_ratio + test_ratio)), random_state=random_seed
            )
        elif valid_ratio > 0:
            valid_data = temp_data
            test_data = []
        elif test_ratio > 0:
            test_data = temp_data
            valid_data = []
        else: # Only train ratio
            train_data = data
            valid_data = []
            test_data = []


        logger.info(f"Split dataset: {len(train_data)} train, {len(valid_data)} validation, {len(test_data)} test")
        return train_data, valid_data, test_data

    def detect_and_fix_problems(self, data: List[Dict[str, Any]], template_name: Optional[str] = None):
        """Auto-detect and fix common problems in the dataset."""
        if not data:
            raise ValueError("No data to analyze.")

        logger.info("Detecting and fixing problems in the dataset...")
        fixed_count = 0
        problems_found = []

        template_structure = DATASET_TEMPLATES.get(template_name, {}).get("structure", {})

        for i, item in enumerate(data):
            item_fixed = False

            # 1. Check for missing required fields based on template
            if template_structure:
                def _check_structure(structure, current_data, path=""):
                    nonlocal item_fixed
                    for key, value in structure.items():
                        full_path = f"{path}.{key}" if path else key
                        if isinstance(value, dict):
                            if key not in current_data or not isinstance(current_data[key], dict):
                                current_data[key] = {}
                                item_fixed = True
                                problems_found.append(f"Record {i+1}: Missing or invalid structure for '{full_path}'. Initialized as empty object.")
                            _check_structure(value, current_data[key], full_path)
                        elif isinstance(value, list) and value and isinstance(value[0], dict):
                            if key not in current_data or not isinstance(current_data[key], list):
                                current_data[key] = [{}] # Initialize with one empty dict for list of objects
                                item_fixed = True
                                problems_found.append(f"Record {i+1}: Missing or invalid structure for '{full_path}'. Initialized as empty list.")
                            else:
                                # Ensure enough elements if template specifies
                                while len(current_data[key]) < len(value):
                                    current_data[key].append({})
                                    item_fixed = True
                                    problems_found.append(f"Record {i+1}: Added missing element to list '{full_path}'.")
                                for j, sub_structure in enumerate(value):
                                    if j < len(current_data[key]):
                                        _check_structure(sub_structure, current_data[key][j], f"{full_path}[{j}]")
                        elif key not in current_data or current_data[key] is None or (isinstance(current_data[key], str) and not current_data[key].strip()):
                            # Missing or empty field, attempt to fill with default
                            if isinstance(value, str):
                                current_data[key] = f"<AUTO_FILLED_{key}>"
                            elif isinstance(value, int):
                                current_data[key] = 0
                            elif isinstance(value, list):
                                current_data[key] = []
                            elif isinstance(value, bool):
                                current_data[key] = False
                            else:
                                current_data[key] = "" # Default to empty string
                            item_fixed = True
                            problems_found.append(f"Record {i+1}: Missing or empty field '{full_path}'. Auto-filled.")

                _check_structure(template_structure, item)

            # 2. Convert None values to appropriate empty types
            def _fix_nones(obj):
                nonlocal item_fixed
                if isinstance(obj, dict):
                    for k, v in list(obj.items()):
                        if v is None:
                            # Try to infer type from template or existing data if available
                            inferred_type = None
                            if template_structure: # More complex path handling needed for nested templates
                                pass # For now, rely on general inference
                            elif self.statistics and k in self.statistics.get("field_types", {}):
                                inferred_type = self.statistics["field_types"][k]

                            if inferred_type == "string":
                                obj[k] = ""
                            elif inferred_type == "number":
                                obj[k] = 0
                            elif inferred_type == "boolean":
                                obj[k] = False
                            elif inferred_type == "array":
                                obj[k] = []
                            elif inferred_type == "object":
                                obj[k] = {}
                            else:
                                obj[k] = "" # Default to empty string
                            item_fixed = True
                            problems_found.append(f"Record {i+1}: Field '{k}' had None value. Auto-converted to empty/default.")
                        else:
                            _fix_nones(v)
                elif isinstance(obj, list):
                    for j, sub_item in enumerate(obj):
                        if sub_item is None:
                            obj[j] = "" # Default list item to empty string
                            item_fixed = True
                            problems_found.append(f"Record {i+1}: List item at index {j} had None value. Auto-converted to empty string.")
                        else:
                            _fix_nones(sub_item)
            _fix_nones(item)

            # 3. Remove empty dictionary/list fields if they are not part of the template structure
            # This is a more aggressive fix, might need user confirmation
            def _clean_empty_structures(obj, template_part=None):
                nonlocal item_fixed
                if isinstance(obj, dict):
                    keys_to_delete = []
                    for k, v in list(obj.items()):
                        if isinstance(v, dict) and not v and (template_part is None or k not in template_part or not isinstance(template_part.get(k), dict)):
                            keys_to_delete.append(k)
                            item_fixed = True
                            problems_found.append(f"Record {i+1}: Removed empty dictionary field '{k}'.")
                        elif isinstance(v, list) and not v and (template_part is None or k not in template_part or not isinstance(template_part.get(k), list)):
                            keys_to_delete.append(k)
                            item_fixed = True
                            problems_found.append(f"Record {i+1}: Removed empty list field '{k}'.")
                        else:
                            _clean_empty_structures(v, template_part.get(k) if template_part and k in template_part else None)
                    for k in keys_to_delete:
                        del obj[k]
                elif isinstance(obj, list):
                    for sub_item in obj:
                        _clean_empty_structures(sub_item, template_part[0] if template_part and len(template_part) > 0 else None)
            _clean_empty_structures(item, template_structure)


            if item_fixed:
                fixed_count += 1
        
        logger.info(f"Fixed issues in {fixed_count} records.")
        return problems_found

    def augment_data(self, data: List[Dict[str, Any]], augmentation_factor=2, methods=None, template_name: Optional[str] = None):
        """Augment dataset using various techniques."""
        if not data:
            raise ValueError("No data to augment.")

        if augmentation_factor <= 1:
            logger.warning("Augmentation factor must be > 1, skipping augmentation.")
            return data

        if not methods:
            methods = ["shuffle"]
            if NLTK_AVAILABLE:
                methods.append("synonym")
            if FAKER_AVAILABLE:
                methods.append("rephrase")

        if not methods:
            raise ValueError("No augmentation methods available. Please install 'nltk' and/or 'faker' for more options.")

        original_data = data.copy()
        augmented_data = []

        logger.info(f"Augmenting dataset (factor: {augmentation_factor})...")

        for item in tqdm(original_data, desc="Augmenting records"):
            for _ in range(augmentation_factor - 1):
                # Choose a random augmentation method
                method = random.choice(methods)
                new_item = self._augment_item(item, method, template_name)
                augmented_data.append(new_item)

        data.extend(augmented_data)
        logger.info(f"Added {len(augmented_data)} augmented records.")
        return data

    def _augment_item(self, item, method, template_name):
        """Apply augmentation to a single record."""
        new_item = json.loads(json.dumps(item)) # Deep copy to avoid modifying original

        template_structure = DATASET_TEMPLATES.get(template_name, {}).get("structure", {})

        def _process_text_fields(structure, current_data):
            for key, value in structure.items():
                if isinstance(value, dict):
                    if key in current_data and isinstance(current_data[key], dict):
                        _process_text_fields(value, current_data[key])
                elif isinstance(value, list) and value and isinstance(value[0], dict):
                    if key in current_data and isinstance(current_data[key], list):
                        for sub_item in current_data[key]:
                            _process_text_fields(value[0], sub_item) # Apply to each item in list
                elif isinstance(current_data.get(key), str) and len(current_data[key].split()) > 3:
                    # Apply augmentation only to string fields that are part of the template
                    if method == "shuffle":
                        words = current_data[key].split()
                        if len(words) > 6:
                            # Don't shuffle the first and last 2 words
                            middle = words[2:-2]
                            random.shuffle(middle)
                            current_data[key] = " ".join(words[:2] + middle + words[-2:])
                    elif method == "synonym" and NLTK_AVAILABLE:
                        current_data[key] = self._replace_with_synonyms(current_data[key])
                    elif method == "rephrase" and FAKER_AVAILABLE:
                        current_data[key] = self._rephrase_text(current_data[key])
        
        # If no template is selected, try to augment all top-level string fields
        if not template_structure:
            for key, value in new_item.items():
                if isinstance(value, str) and len(value.split()) > 3:
                    if method == "shuffle":
                        words = value.split()
                        if len(words) > 6:
                            middle = words[2:-2]
                            random.shuffle(middle)
                            new_item[key] = " ".join(words[:2] + middle + words[-2:])
                    elif method == "synonym" and NLTK_AVAILABLE:
                        new_item[key] = self._replace_with_synonyms(value)
                    elif method == "rephrase" and FAKER_AVAILABLE:
                        new_item[key] = self._rephrase_text(value)
        else:
            _process_text_fields(template_structure, new_item)

        return new_item

    def _get_synonyms(self, word):
        """Helper to get synonyms using NLTK WordNet."""
        synonyms = []
        for syn in wordnet.synsets(word):
            if syn is not None:
                for lemma in syn.lemmas():
                    if lemma.name() != word and "_" not in lemma.name():
                        synonyms.append(lemma.name().replace('_', ' ')) # Replace underscores for readability
        return list(set(synonyms))

    def _replace_with_synonyms(self, text):
        """Replace some words in text with synonyms."""
        if not text or len(text.split()) < 4:
            return text

        words = word_tokenize(text) # Use NLTK tokenizer
        new_words = []
        for word in words:
            if random.random() < 0.2:  # 20% chance to replace
                synonyms = self._get_synonyms(word.lower())
                if synonyms:
                    new_words.append(random.choice(synonyms))
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
        return " ".join(new_words)

    def _rephrase_text(self, text):
        """Rephrase text using Faker (simple rephrasing)."""
        if not text or len(text) < 10:
            return text

        # Faker doesn't "rephrase" directly, but can generate similar content
        # We'll use a heuristic: generate a new paragraph/sentence of similar length
        # This is a simple approximation of rephrasing for offline use.
        if random.random() < 0.5: # 50% chance to rephrase
            if len(text.split()) > 10: # If it's a longer text, use paragraph
                return faker.paragraph(nb_sentences=random.randint(1,3), variable_nb_sentences=True)
            else: # Otherwise, use sentence
                return faker.sentence(nb_words=random.randint(5,15), variable_nb_words=True)
        return text


    def visualize_data(self, data: List[Dict[str, Any]], output_path=None, selected_field=None):
        """Create visualizations of the dataset."""
        if not VISUALIZATION_AVAILABLE:
            raise ImportError("matplotlib and seaborn libraries not available for visualization.")

        if not data:
            raise ValueError("No data to visualize.")

        logger.info("Generating dataset visualizations...")

        # Convert to DataFrame for easier visualization
        df = pd.DataFrame(data)

        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten() # Flatten for easy iteration

        # 1. Record count
        axes[0].setTitle("Dataset Overview")
        axes[0].bar(["Records"], [len(data)], color="steelblue")
        axes[0].set_ylabel("Count")
        axes[0].text(0, len(data), str(len(data)), ha='center', va='bottom')


        # 2. Field value lengths (for string fields)
        axes[1].setTitle("Average Field Lengths (String Fields)")
        string_fields = [col for col in df.columns if df[col].dtype == object and df[col].apply(lambda x: isinstance(x, str)).any()]
        
        if string_fields:
            lengths = df[string_fields].applymap(lambda x: len(str(x)) if x is not None else 0).mean().sort_values(ascending=False)
            sns.barplot(x=lengths.values, y=lengths.index, ax=axes[1], palette="viridis")
            axes[1].set_xlabel("Average Length")
        else:
            axes[1].text(0.5, 0.5, "No string fields for length analysis", horizontalalignment='center',
                    verticalalignment='center', transform=axes[1].transAxes)


        # 3. Missing values
        axes[2].setTitle("Missing Values Count")
        missing_counts = df.isnull().sum()
        missing_fields = missing_counts[missing_counts > 0]

        if not missing_fields.empty:
            sns.barplot(x=missing_fields.values, y=missing_fields.index, ax=axes[2], palette="magma")
            axes[2].set_xlabel("Count of Missing Values")
        else:
            axes[2].text(0.5, 0.5, "No missing values found", horizontalalignment='center',
                    verticalalignment='center', transform=axes[2].transAxes)


        # 4. Data distribution (for a selected field or first suitable field)
        axes[3].setTitle(f"Value Distribution for '{selected_field or 'N/A'}'")
        if selected_field and selected_field in df.columns:
            if df[selected_field].dtype == object: # Categorical/string
                value_counts = df[selected_field].value_counts().head(10)
                if len(value_counts) > 0:
                    sns.barplot(x=value_counts.values, y=value_counts.index, ax=axes[3], palette="cividis")
                    axes[3].set_xlabel("Count")
                else:
                    axes[3].text(0.5, 0.5, f"No data for '{selected_field}' distribution", horizontalalignment='center',
                            verticalalignment='center', transform=axes[3].transAxes)
            elif pd.api.types.is_numeric_dtype(df[selected_field]): # Numerical
                sns.histplot(df[selected_field].dropna(), kde=True, ax=axes[3], color="orange")
                axes[3].set_xlabel(selected_field)
                axes[3].set_ylabel("Frequency")
            else:
                axes[3].text(0.5, 0.5, f"Cannot plot distribution for '{selected_field}' (unsupported type)", horizontalalignment='center',
                        verticalalignment='center', transform=axes[3].transAxes)
        else:
            axes[3].text(0.5, 0.5, "Select a field for distribution analysis", horizontalalignment='center',
                    verticalalignment='center', transform=axes[3].transAxes)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path)
            logger.info(f"Saved visualization to {output_path}")
        else:
            # If not saving, ensure the plot is displayed if not running in a non-interactive backend
            pass # Matplotlib will handle showing if a backend is active

        plt.close(fig) # Close the figure to free memory

    def anonymize_data(self, data: List[Dict[str, Any]], fields_to_anonymize=None):
        """Anonymize sensitive fields in the dataset."""
        if not FAKER_AVAILABLE:
            raise ImportError("Faker library not available for anonymization.")

        if not data:
            raise ValueError("No data to anonymize.")

        # If no fields specified, try to auto-detect potentially sensitive fields
        if not fields_to_anonymize:
            sensitive_patterns = {
                "email": re.compile(r'email|mail|e-mail', re.I),
                "name": re.compile(r'name|username|user|author', re.I),
                "phone": re.compile(r'phone|tel|mobile', re.I),
                "address": re.compile(r'address|location|city|state|zip|postal', re.I),
                "id": re.compile(r'id$|identifier|account', re.I),
                "ip": re.compile(r'ip$|ip_address', re.I),
                "ssn": re.compile(r'ssn|social|security', re.I),
                "credit_card": re.compile(r'credit|card|payment', re.I),
                "dob": re.compile(r'birth|dob|birthday', re.I),
            }

            fields_to_anonymize = {}
            if data:
                sample = data[0]
                for field in sample.keys():
                    field_lower = field.lower()
                    for field_type, pattern in sensitive_patterns.items():
                        if pattern.search(field_lower):
                            fields_to_anonymize[field] = field_type
                            break

        if not fields_to_anonymize:
            logger.warning("No fields to anonymize detected.")
            return data

        logger.info("Anonymizing sensitive data...")
        anonymized_count = 0

        for item in tqdm(data, desc="Anonymizing records"):
            for field, field_type in fields_to_anonymize.items():
                if field in item and item[field] is not None and item[field] != "":
                    original_value = item[field]
                    if field_type == "email":
                        item[field] = faker.email()
                    elif field_type == "name":
                        item[field] = faker.name()
                    elif field_type == "phone":
                        item[field] = faker.phone_number()
                    elif field_type == "address":
                        item[field] = faker.address()
                    elif field_type == "id":
                        item[field] = str(uuid.uuid4())
                    elif field_type == "ip":
                        item[field] = faker.ipv4()
                    elif field_type == "ssn":
                        item[field] = faker.ssn()
                    elif field_type == "credit_card":
                        item[field] = faker.credit_card_number()
                    elif field_type == "dob":
                        item[field] = faker.date_of_birth().isoformat()
                    else:
                        item[field] = f"ANONYMIZED_{field}"
                    anonymized_count += 1
        logger.info(f"Anonymized {anonymized_count} values across {len(fields_to_anonymize)} fields.")
        return data

# --- GUI Components and Main Window ---

class CustomValidationRuleDialog(QDialog):
    """Dialog for defining custom validation rules."""
    def __init__(self, current_rules: List[Dict], all_fields: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Custom Validation Rules")
        self.setGeometry(200, 200, 600, 400)
        self.setModal(True)

        self.all_fields = all_fields
        self.rules = current_rules if current_rules else []
        self.rule_widgets = [] # To keep track of dynamically created rule widgets

        self._init_ui()
        self._load_existing_rules()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        description_label = QLabel("Define custom rules for validating your dataset. Rules are applied sequentially.")
        description_label.setWordWrap(True)
        main_layout.addWidget(description_label)

        self.rules_scroll_area = QScrollArea()
        self.rules_scroll_area.setWidgetResizable(True)
        self.rules_container = QWidget()
        self.rules_layout = QVBoxLayout(self.rules_container)
        self.rules_layout.setAlignment(Qt.AlignTop)
        self.rules_scroll_area.setWidget(self.rules_container)
        main_layout.addWidget(self.rules_scroll_area)

        add_rule_button = QPushButton("Add New Rule")
        add_rule_button.clicked.connect(self._add_rule_widget)
        main_layout.addWidget(add_rule_button)

        button_box = QHBoxLayout()
        ok_button = QPushButton("Save Rules")
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        button_box.addWidget(ok_button)
        button_box.addWidget(cancel_button)
        main_layout.addLayout(button_box)

    def _add_rule_widget(self, rule_data: Optional[Dict] = None):
        rule_group = QGroupBox()
        rule_layout = QFormLayout(rule_group)

        field_combo = QComboBox()
        field_combo.addItem("ALL_FIELDS") # Option to apply to all fields
        field_combo.addItems(sorted(self.all_fields))
        rule_layout.addRow("Apply to Field:", field_combo)

        rule_type_combo = QComboBox()
        rule_type_combo.addItems(["Required", "Type Check", "Regex Match", "Min Length", "Max Length", "Min Value", "Max Value", "Custom Python"])
        rule_layout.addRow("Rule Type:", rule_type_combo)

        # Rule specific inputs
        value_input = QLineEdit()
        value_input.setPlaceholderText("e.g., 'string', '^[A-Z]', '10', 'lambda x: x > 0'")
        rule_layout.addRow("Value/Pattern:", value_input)

        # Description for custom python rule
        custom_python_desc = QLabel("For 'Custom Python', enter a lambda function (e.g., 'lambda x: x > 0'). 'x' is the field value.")
        custom_python_desc.setWordWrap(True)
        custom_python_desc.setVisible(False)
        rule_layout.addRow(custom_python_desc)

        def update_value_input_visibility():
            rule_type = rule_type_combo.currentText()
            if rule_type == "Required":
                value_input.setVisible(False)
                custom_python_desc.setVisible(False)
            elif rule_type == "Custom Python":
                value_input.setVisible(True)
                custom_python_desc.setVisible(True)
                value_input.setPlaceholderText("e.g., 'lambda x: x > 0'")
            elif rule_type == "Type Check":
                value_input.setVisible(True)
                custom_python_desc.setVisible(False)
                value_input.setPlaceholderText("e.g., 'string', 'number', 'boolean', 'list', 'dict'")
            elif rule_type in ["Min Length", "Max Length", "Min Value", "Max Value"]:
                value_input.setVisible(True)
                custom_python_desc.setVisible(False)
                value_input.setPlaceholderText("Enter a number")
            else: # Regex Match
                value_input.setVisible(True)
                custom_python_desc.setVisible(False)
                value_input.setPlaceholderText("Enter a regex pattern (e.g., '^[A-Z]')")

        rule_type_combo.currentTextChanged.connect(update_value_input_visibility)
        update_value_input_visibility() # Initial call

        remove_button = QPushButton("Remove Rule")
        remove_button.clicked.connect(lambda: self._remove_rule_widget(rule_group))
        rule_layout.addRow(remove_button)

        self.rules_layout.addWidget(rule_group)
        self.rule_widgets.append({
            "group": rule_group,
            "field_combo": field_combo,
            "rule_type_combo": rule_type_combo,
            "value_input": value_input,
            "custom_python_desc": custom_python_desc
        })

        if rule_data:
            field_combo.setCurrentText(rule_data.get("field", "ALL_FIELDS"))
            rule_type_combo.setCurrentText(rule_data.get("type", "Required"))
            value_input.setText(str(rule_data.get("value", "")))
            update_value_input_visibility() # Update visibility after setting text

    def _remove_rule_widget(self, group_to_remove: QGroupBox):
        group_to_remove.deleteLater()
        self.rule_widgets = [w for w in self.rule_widgets if w["group"] != group_to_remove]

    def _load_existing_rules(self):
        for rule_data in self.rules:
            self._add_rule_widget(rule_data)

    def get_rules(self) -> List[Dict]:
        """Collects rules from the UI widgets."""
        collected_rules = []
        for widget_set in self.rule_widgets:
            rule = {
                "field": widget_set["field_combo"].currentText(),
                "type": widget_set["rule_type_combo"].currentText(),
                "value": widget_set["value_input"].text()
            }
            collected_rules.append(rule)
        self.rules = collected_rules # Update internal list
        return self.rules

class DragDropTableWidget(QTableWidget):
    """TableWidget with drag and drop support for files."""
    file_dropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                self.file_dropped.emit(file_path)
                break  # Just take the first file


class CustomComboBox(QComboBox):
    """ComboBox with descriptions in tooltips."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setToolTip("Click to show options")
        self.descriptions = {}

    def addItemWithDescription(self, text, description):
        self.addItem(text)
        self.descriptions[text] = description

    def showPopup(self):
        super().showPopup()

    def hidePopup(self):
        super().hidePopup()

    def event(self, event):
        if event.type() == event.Type.ToolTip:
            index = self.view().indexAt(event.pos())
            if index.isValid():
                text = self.itemText(index.row())
                if text in self.descriptions:
                    QToolTip.showText(event.globalPos(), self.descriptions[text])
                    return True
        return super().event(event)


class MainWindow(QMainWindow):
    """Main application window for the Dataset Generator GUI."""

    def __init__(self):
        super().__init__()

        self.generator = DatasetGenerator() # Instantiate the core logic
        self.input_data = [] # Data currently loaded (e.g., from input file)
        self.output_data = [] # Data after processing/conversion/generation
        self.current_template = None
        self.field_mapping = {}
        self.custom_validation_rules = [] # Stores custom rules

        self.setWindowTitle("Dataset Generator for LLM Fine-Tuning (Offline)")
        self.setGeometry(100, 100, 1200, 800)
        self.setWindowIcon(self._get_icon())

        # Initialize worker thread
        self.worker = ProcessingWorker()
        self.worker.status_signal.connect(self.update_status)
        self.worker.error_signal.connect(self.show_error)
        self.worker.progress_signal.connect(self.update_progress_bar)
        self.worker.finished_signal.connect(self.process_finished)

        # Set up UI elements
        self._create_actions()
        self._create_menu_bar()
        self._create_toolbar()
        self._create_status_bar()
        self._create_central_widget()

        # Apply dark theme
        self._apply_theme()

        # Load settings
        self._load_settings()

        # Final setup
        self.tabs.setCurrentIndex(0)

    def _get_icon(self):
        """Create an application icon."""
        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw a simple icon - dataset symbol
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(41, 128, 185))  # Blue
        painter.drawEllipse(5, 5, 54, 54)

        painter.setBrush(QColor(255, 255, 255))  # White
        painter.drawEllipse(15, 15, 34, 34)

        painter.setBrush(QColor(41, 128, 185))  # Blue
        painter.drawEllipse(25, 25, 14, 14)

        painter.end()

        return QIcon(pixmap)

    def _apply_theme(self):
        """Apply a modern dark theme to the application."""
        dark_palette = QPalette()

        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)

        self.setPalette(dark_palette)

    def _create_actions(self):
        """Create application actions."""
        # File actions
        self.open_action = QAction("&Open Dataset...", self)
        self.open_action.setShortcut("Ctrl+O")
        self.open_action.triggered.connect(self.open_dataset)

        self.save_action = QAction("&Save Dataset...", self)
        self.save_action.setShortcut("Ctrl+S")
        self.save_action.triggered.connect(self.save_dataset)

        self.export_action = QAction("&Export as...", self)
        self.export_action.setShortcut("Ctrl+E")
        self.export_action.triggered.connect(self.export_dataset)

        self.exit_action = QAction("E&xit", self)
        self.exit_action.setShortcut("Alt+F4")
        self.exit_action.triggered.connect(self.close)

        # Tools actions
        self.validate_action = QAction("&Validate Dataset", self)
        self.validate_action.triggered.connect(self.validate_dataset_menu)

        self.fix_action = QAction("Auto-&Fix Issues", self)
        self.fix_action.triggered.connect(self.fix_dataset_menu)

        self.augment_action = QAction("&Augment Data", self)
        self.augment_action.triggered.connect(self.augment_dataset)

        self.anonymize_action = QAction("&Anonymize Data", self)
        self.anonymize_action.triggered.connect(self.anonymize_dataset)

        self.split_action = QAction("&Split Dataset", self)
        self.split_action.triggered.connect(self.split_dataset)

        self.visualize_action = QAction("&Visualize Data", self)
        self.visualize_action.triggered.connect(self.visualize_dataset_menu)

        # Help actions
        self.about_action = QAction("&About", self)
        self.about_action.triggered.connect(self.show_about)

        self.help_action = QAction("&Help", self)
        self.help_action.triggered.connect(self.show_help)

    def _create_menu_bar(self):
        """Create the application menu bar."""
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.export_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        # Tools menu
        tools_menu = menu_bar.addMenu("&Tools")
        tools_menu.addAction(self.validate_action)
        tools_menu.addAction(self.fix_action)
        tools_menu.addAction(self.augment_action)
        tools_menu.addAction(self.anonymize_action)
        tools_menu.addAction(self.split_action)
        tools_menu.addAction(self.visualize_action)

        # Help menu
        help_menu = menu_bar.addMenu("&Help")
        help_menu.addAction(self.about_action)
        help_menu.addAction(self.help_action)

    def _create_toolbar(self):
        """Create the main toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        toolbar.addAction(self.open_action)
        toolbar.addAction(self.save_action)
        toolbar.addAction(self.export_action)
        toolbar.addSeparator()
        toolbar.addAction(self.validate_action)
        toolbar.addAction(self.fix_action)
        toolbar.addAction(self.visualize_action)

    def _create_status_bar(self):
        """Create the status bar."""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(150)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def _create_central_widget(self):
        """Create the central widget with tabs."""
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)

        # Create tabs for different operations
        self.convert_tab = self._create_convert_tab()
        self.create_tab = self._create_create_tab()
        self.edit_tab = self._create_edit_tab()
        self.validate_tab = self._create_validate_tab()
        self.visualize_tab = self._create_visualize_tab()

        # Add tabs to the widget
        self.tabs.addTab(self.convert_tab, "Convert")
        self.tabs.addTab(self.create_tab, "Create")
        self.tabs.addTab(self.edit_tab, "Edit")
        self.tabs.addTab(self.validate_tab, "Validate")
        self.tabs.addTab(self.visualize_tab, "Visualize")

        # Connect tab changed signal
        self.tabs.currentChanged.connect(self._on_tab_changed)

        layout.addWidget(self.tabs)
        self.setCentralWidget(central_widget)

    def _create_convert_tab(self):
        """Create the Convert tab for format conversion."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        # Header section
        header_label = QLabel("Convert Dataset Formats")
        header_label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(header_label)

        description_label = QLabel(
            "Import a dataset from one format and convert it to another format. "
            "You can also map fields to a specific template structure."
        )
        description_label.setWordWrap(True)
        layout.addWidget(description_label)

        # Input section
        input_group = QGroupBox("1. Select Input Dataset")
        input_layout = QVBoxLayout(input_group)

        file_layout = QHBoxLayout()
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText("Path to input file (or drag and drop here)...")
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_input_file)
        file_layout.addWidget(self.input_path_edit)
        file_layout.addWidget(browse_button)
        input_layout.addLayout(file_layout)

        format_layout = QHBoxLayout()
        format_label = QLabel("Format:")
        self.input_format_combo = CustomComboBox()
        for format_name, description in INPUT_FORMATS.items():
            self.input_format_combo.addItemWithDescription(format_name, description)
        format_layout.addWidget(format_label)
        format_layout.addWidget(self.input_format_combo)
        input_layout.addLayout(format_layout)

        load_button = QPushButton("Load Dataset")
        load_button.clicked.connect(self._load_dataset)
        input_layout.addWidget(load_button)

        layout.addWidget(input_group)

        # Template section
        template_group = QGroupBox("2. Select Template Format")
        template_layout = QVBoxLayout(template_group)

        self.template_combo = CustomComboBox()
        for template_name, template_info in DATASET_TEMPLATES.items():
            self.template_combo.addItemWithDescription(
                template_name, template_info["description"]
            )
        template_layout.addWidget(self.template_combo)

        self.template_info_label = QLabel("Select a template to see more information.")
        self.template_info_label.setWordWrap(True)
        template_layout.addWidget(self.template_info_label)

        self.template_combo.currentTextChanged.connect(self._update_template_info)

        template_layout.addWidget(QLabel("Preview of template structure:"))
        self.template_structure_text = QTextEdit()
        self.template_structure_text.setReadOnly(True)
        self.template_structure_text.setMaximumHeight(120)
        template_layout.addWidget(self.template_structure_text)

        template_layout.addWidget(QLabel("Note: Select 'custom' to define your own structure."))

        layout.addWidget(template_group)

        # Output section
        output_group = QGroupBox("3. Set Output Options")
        output_layout = QVBoxLayout(output_group)

        self.output_format_combo = CustomComboBox()
        for format_name, description in OUTPUT_FORMATS.items():
            self.output_format_combo.addItemWithDescription(format_name, description)
        output_layout.addWidget(QLabel("Output format:"))
        output_layout.addWidget(self.output_format_combo)

        # Output file path
        output_file_layout = QHBoxLayout()
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("Path to output file...")
        output_browse_button = QPushButton("Browse...")
        output_browse_button.clicked.connect(self._browse_output_file)
        output_file_layout.addWidget(self.output_path_edit)
        output_file_layout.addWidget(output_browse_button)
        output_layout.addLayout(output_file_layout)

        layout.addWidget(output_group)

        # Actions section
        actions_layout = QHBoxLayout()

        self.map_fields_button = QPushButton("Map Fields")
        self.map_fields_button.clicked.connect(self._show_field_mapping_dialog)
        self.map_fields_button.setEnabled(False)
        actions_layout.addWidget(self.map_fields_button)

        self.preview_button = QPushButton("Preview Result")
        self.preview_button.clicked.connect(self._preview_converted_data)
        self.preview_button.setEnabled(False)
        actions_layout.addWidget(self.preview_button)

        self.convert_button = QPushButton("Convert & Save")
        self.convert_button.clicked.connect(self._convert_dataset)
        self.convert_button.setEnabled(False)
        actions_layout.addWidget(self.convert_button)

        layout.addLayout(actions_layout)

        # Add a table for preview
        self.preview_label = QLabel("Dataset Preview (first 100 records):")
        layout.addWidget(self.preview_label)

        self.preview_table = DragDropTableWidget()
        self.preview_table.file_dropped.connect(self._handle_dropped_file)
        self.preview_table.setEditTriggers(QTableWidget.NoEditTriggers) # Not editable in preview
        layout.addWidget(self.preview_table)

        # Stretch to fill space
        layout.addStretch()

        return tab

    def _create_create_tab(self):
        """Create the Create tab for dataset creation."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        # Header section
        header_label = QLabel("Create New Dataset")
        header_label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(header_label)

        description_label = QLabel(
            "Create a new dataset from scratch by defining its structure and content. "
            "You can create individual records or generate them automatically using templates and local data generation."
        )
        description_label.setWordWrap(True)
        layout.addWidget(description_label)

        # Template selection section
        template_group = QGroupBox("1. Select Dataset Template")
        template_layout = QVBoxLayout(template_group)

        self.create_template_combo = CustomComboBox()
        for template_name, template_info in DATASET_TEMPLATES.items():
            self.create_template_combo.addItemWithDescription(
                template_name, template_info["description"]
            )
        template_layout.addWidget(self.create_template_combo)

        self.create_template_info_label = QLabel("Select a template to see more information.")
        self.create_template_info_label.setWordWrap(True)
        template_layout.addWidget(self.create_template_info_label)

        self.create_template_combo.currentTextChanged.connect(self._update_create_template_info)

        layout.addWidget(template_group)

        # Creation options
        options_group = QGroupBox("2. Generation Options")
        options_layout = QVBoxLayout(options_group)

        # Number of records
        record_layout = QHBoxLayout()
        record_layout.addWidget(QLabel("Number of records to generate:"))
        self.num_records_spin = QSpinBox()
        self.num_records_spin.setRange(1, 100000) # Increased max for larger datasets
        self.num_records_spin.setValue(10)
        record_layout.addWidget(self.num_records_spin)
        record_layout.addStretch()
        options_layout.addLayout(record_layout)

        # Generation options
        self.auto_generate_check = QCheckBox("Auto-generate content (using Faker if available)")
        self.auto_generate_check.setChecked(True)
        options_layout.addWidget(self.auto_generate_check)

        self.random_seed_check = QCheckBox("Use random seed for reproducibility")
        options_layout.addWidget(self.random_seed_check)

        seed_layout = QHBoxLayout()
        seed_layout.addWidget(QLabel("Random seed:"))
        self.random_seed_spin = QSpinBox()
        self.random_seed_spin.setRange(0, 999999)
        self.random_seed_spin.setValue(42)
        self.random_seed_spin.setEnabled(False)
        seed_layout.addWidget(self.random_seed_spin)
        seed_layout.addStretch()
        options_layout.addLayout(seed_layout)

        self.random_seed_check.toggled.connect(self.random_seed_spin.setEnabled)

        layout.addWidget(options_group)

        # Creation action
        create_button = QPushButton("Generate Dataset")
        create_button.clicked.connect(self._generate_dataset)
        layout.addWidget(create_button)

        # Data view
        self.created_data_label = QLabel("Generated Data Preview (first 100 records):")
        layout.addWidget(self.created_data_label)

        self.created_data_table = QTableWidget()
        self.created_data_table.setEditTriggers(QTableWidget.NoEditTriggers) # Not editable here
        layout.addWidget(self.created_data_table)

        # Export section
        export_group = QGroupBox("3. Export Generated Dataset")
        export_layout = QVBoxLayout(export_group)

        self.create_format_combo = CustomComboBox()
        for format_name, description in OUTPUT_FORMATS.items():
            self.create_format_combo.addItemWithDescription(format_name, description)
        export_layout.addWidget(QLabel("Output format:"))
        export_layout.addWidget(self.create_format_combo)

        # Output file path
        create_file_layout = QHBoxLayout()
        self.create_output_path_edit = QLineEdit()
        self.create_output_path_edit.setPlaceholderText("Path to output file...")
        create_browse_button = QPushButton("Browse...")
        create_browse_button.clicked.connect(self._browse_create_output_file)
        create_file_layout.addWidget(self.create_output_path_edit)
        create_file_layout.addWidget(create_browse_button)
        export_layout.addLayout(create_file_layout)

        export_button = QPushButton("Export Dataset")
        export_button.clicked.connect(self._export_created_dataset)
        export_layout.addWidget(export_button)

        layout.addWidget(export_group)

        # Stretch to fill space
        layout.addStretch()

        return tab

    def _create_edit_tab(self):
        """Create the Edit tab for modifying datasets."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        # Header section
        header_label = QLabel("Edit Dataset")
        header_label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(header_label)

        description_label = QLabel(
            "Load an existing dataset and modify its content. "
            "You can add, edit, delete, or filter records directly in the table."
        )
        description_label.setWordWrap(True)
        layout.addWidget(description_label)

        # Input section
        input_group = QGroupBox("1. Load Dataset to Edit")
        input_layout = QVBoxLayout(input_group)

        file_layout = QHBoxLayout()
        self.edit_path_edit = QLineEdit()
        self.edit_path_edit.setPlaceholderText("Path to input file...")
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_edit_file)
        file_layout.addWidget(self.edit_path_edit)
        file_layout.addWidget(browse_button)
        input_layout.addLayout(file_layout)

        format_layout = QHBoxLayout()
        format_label = QLabel("Format:")
        self.edit_format_combo = CustomComboBox()
        for format_name, description in INPUT_FORMATS.items():
            self.edit_format_combo.addItemWithDescription(format_name, description)
        format_layout.addWidget(format_label)
        format_layout.addWidget(self.edit_format_combo)
        input_layout.addLayout(format_layout)

        load_button = QPushButton("Load Dataset")
        load_button.clicked.connect(self._load_edit_dataset)
        input_layout.addWidget(load_button)

        layout.addWidget(input_group)

        # Editing tools
        tools_group = QGroupBox("2. Editing Tools")
        tools_layout = QHBoxLayout(tools_group)

        add_button = QPushButton("Add Record")
        add_button.clicked.connect(self._add_record)
        tools_layout.addWidget(add_button)

        delete_button = QPushButton("Delete Selected")
        delete_button.clicked.connect(self._delete_selected_records)
        tools_layout.addWidget(delete_button)

        # Search and Filter
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        self.edit_search_input = QLineEdit()
        self.edit_search_input.setPlaceholderText("Type to search/filter...")
        self.edit_search_input.textChanged.connect(self._filter_edit_table)
        search_layout.addWidget(self.edit_search_input)
        tools_layout.addLayout(search_layout)

        layout.addWidget(tools_group)

        # Data display
        self.edit_data_table = QTableWidget()
        self.edit_data_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.edit_data_table.setEditTriggers(QTableWidget.DoubleClicked | QTableWidget.AnyKeyPressed) # Editable
        self.edit_data_table.itemChanged.connect(self._handle_edit_table_item_changed)
        layout.addWidget(self.edit_data_table)

        # Save section
        save_group = QGroupBox("3. Save Changes")
        save_layout = QVBoxLayout(save_group)

        self.edit_output_format_combo = CustomComboBox()
        for format_name, description in OUTPUT_FORMATS.items():
            self.edit_output_format_combo.addItemWithDescription(format_name, description)
        save_layout.addWidget(QLabel("Output format:"))
        save_layout.addWidget(self.edit_output_format_combo)

        # Output file path
        save_file_layout = QHBoxLayout()
        self.edit_output_path_edit = QLineEdit()
        self.edit_output_path_edit.setPlaceholderText("Path to output file...")
        save_browse_button = QPushButton("Browse...")
        save_browse_button.clicked.connect(self._browse_edit_output_file)
        save_file_layout.addWidget(self.edit_output_path_edit)
        save_file_layout.addWidget(save_browse_button)
        save_layout.addLayout(save_file_layout)

        save_button = QPushButton("Save Dataset")
        save_button.clicked.connect(self._save_edited_dataset)
        save_layout.addWidget(save_button)

        layout.addWidget(save_group)

        return tab

    def _create_validate_tab(self):
        """Create the Validate tab for validating datasets."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        # Header section
        header_label = QLabel("Validate Dataset")
        header_label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(header_label)

        description_label = QLabel(
            "Validate a dataset against a template structure or custom rules. "
            "Find and fix issues with your dataset to ensure data quality."
        )
        description_label.setWordWrap(True)
        layout.addWidget(description_label)

        # Input section
        input_group = QGroupBox("1. Select Dataset to Validate")
        input_layout = QVBoxLayout(input_group)

        file_layout = QHBoxLayout()
        self.validate_path_edit = QLineEdit()
        self.validate_path_edit.setPlaceholderText("Path to input file...")
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_validate_file)
        file_layout.addWidget(self.validate_path_edit)
        file_layout.addWidget(browse_button)
        input_layout.addLayout(file_layout)

        format_layout = QHBoxLayout()
        format_label = QLabel("Format:")
        self.validate_format_combo = CustomComboBox()
        for format_name, description in INPUT_FORMATS.items():
            self.validate_format_combo.addItemWithDescription(format_name, description)
        format_layout.addWidget(format_label)
        format_layout.addWidget(self.validate_format_combo)
        input_layout.addLayout(format_layout)

        load_button = QPushButton("Load Dataset")
        load_button.clicked.connect(self._load_validate_dataset)
        input_layout.addWidget(load_button)

        layout.addWidget(input_group)

        # Validation options
        validation_group = QGroupBox("2. Validation Options")
        validation_layout = QVBoxLayout(validation_group)

        # Template selection
        template_layout = QHBoxLayout()
        template_layout.addWidget(QLabel("Validate against template:"))
        self.validate_template_combo = CustomComboBox()
        for template_name, template_info in DATASET_TEMPLATES.items():
            self.validate_template_combo.addItemWithDescription(
                template_name, template_info["description"]
            )
        template_layout.addWidget(self.validate_template_combo)
        validation_layout.addLayout(template_layout)

        # Built-in checks
        self.check_missing_values = QCheckBox("Check for missing values")
        self.check_missing_values.setChecked(True)
        validation_layout.addWidget(self.check_missing_values)

        self.check_field_types = QCheckBox("Validate field types (consistency)")
        self.check_field_types.setChecked(True)
        validation_layout.addWidget(self.check_field_types)

        self.check_duplicate_records = QCheckBox("Check for duplicate records")
        self.check_duplicate_records.setChecked(True)
        validation_layout.addWidget(self.check_duplicate_records)

        # Custom rules
        custom_rules_layout = QHBoxLayout()
        custom_rules_layout.addWidget(QLabel("Custom Rules:"))
        self.edit_custom_rules_button = QPushButton("Edit Custom Rules...")
        self.edit_custom_rules_button.clicked.connect(self._edit_custom_validation_rules)
        custom_rules_layout.addWidget(self.edit_custom_rules_button)
        validation_layout.addLayout(custom_rules_layout)

        layout.addWidget(validation_group)

        # Validation action
        validate_button = QPushButton("Run Validation")
        validate_button.clicked.connect(self._validate_dataset_action)
        layout.addWidget(validate_button)

        # Results display
        results_group = QGroupBox("Validation Results")
        results_layout = QVBoxLayout(results_group)

        self.validation_results_text = QTextEdit()
        self.validation_results_text.setReadOnly(True)
        results_layout.addWidget(self.validation_results_text)

        self.fix_issues_button = QPushButton("Auto-Fix Issues")
        self.fix_issues_button.clicked.connect(self._auto_fix_validation_issues)
        self.fix_issues_button.setEnabled(False)
        results_layout.addWidget(self.fix_issues_button)

        layout.addWidget(results_group)

        # Data preview
        self.validate_data_table = QTableWidget()
        self.validate_data_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.validate_data_table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.validate_data_table)

        # Stretch to fill space
        layout.addStretch()

        return tab

    def _create_visualize_tab(self):
        """Create the Visualize tab for data visualization."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        # Header section
        header_label = QLabel("Visualize Dataset")
        header_label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(header_label)

        description_label = QLabel(
            "Generate visualizations of your dataset to understand its structure and content. "
            "View statistics and summaries of your data."
        )
        description_label.setWordWrap(True)
        layout.addWidget(description_label)

        # Input section
        input_group = QGroupBox("1. Select Dataset to Visualize")
        input_layout = QVBoxLayout(input_group)

        file_layout = QHBoxLayout()
        self.visualize_path_edit = QLineEdit()
        self.visualize_path_edit.setPlaceholderText("Path to input file...")
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_visualize_file)
        file_layout.addWidget(self.visualize_path_edit)
        file_layout.addWidget(browse_button)
        input_layout.addLayout(file_layout)

        format_layout = QHBoxLayout()
        format_label = QLabel("Format:")
        self.visualize_format_combo = CustomComboBox()
        for format_name, description in INPUT_FORMATS.items():
            self.visualize_format_combo.addItemWithDescription(format_name, description)
        format_layout.addWidget(format_label)
        format_layout.addWidget(self.visualize_format_combo)
        input_layout.addLayout(format_layout)

        load_button = QPushButton("Load Dataset")
        load_button.clicked.connect(self._load_visualize_dataset)
        input_layout.addWidget(load_button)

        layout.addWidget(input_group)

        # Visualization options
        viz_options_group = QGroupBox("2. Visualization Options")
        viz_options_layout = QVBoxLayout(viz_options_group)

        # Visualization types
        self.viz_overview_check = QCheckBox("Dataset Overview (Record Count)")
        self.viz_overview_check.setChecked(True)
        viz_options_layout.addWidget(self.viz_overview_check)

        self.viz_field_lengths_check = QCheckBox("Field Lengths Analysis (String Fields)")
        self.viz_field_lengths_check.setChecked(True)
        viz_options_layout.addWidget(self.viz_field_lengths_check)

        self.viz_missing_values_check = QCheckBox("Missing Values Analysis")
        self.viz_missing_values_check.setChecked(True)
        viz_options_layout.addWidget(self.viz_missing_values_check)

        self.viz_distribution_check = QCheckBox("Value Distribution (for selected field)")
        self.viz_distribution_check.setChecked(True)
        viz_options_layout.addWidget(self.viz_distribution_check)

        # Field selection for distribution
        field_layout = QHBoxLayout()
        field_layout.addWidget(QLabel("Select field for distribution:"))
        self.viz_field_combo = QComboBox()
        field_layout.addWidget(self.viz_field_combo)
        viz_options_layout.addLayout(field_layout)

        layout.addWidget(viz_options_group)

        # Visualization action
        viz_button = QPushButton("Generate Visualizations")
        viz_button.clicked.connect(self._generate_visualizations)
        layout.addWidget(viz_button)

        # Stats display
        stats_group = QGroupBox("Dataset Statistics")
        stats_layout = QVBoxLayout(stats_group)

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text)

        layout.addWidget(stats_group)

        # Visualization display (placeholder)
        self.viz_placeholder = QLabel("Visualizations will appear here after generation.")
        self.viz_placeholder.setAlignment(Qt.AlignCenter)
        self.viz_placeholder.setMinimumHeight(300)
        self.viz_placeholder.setStyleSheet("background-color: #2a2a2a; border: 1px solid #444;")
        layout.addWidget(self.viz_placeholder)

        # Export options
        export_layout = QHBoxLayout()
        self.save_viz_button = QPushButton("Save Visualizations")
        self.save_viz_button.clicked.connect(self._save_visualizations)
        self.save_viz_button.setEnabled(False)
        export_layout.addWidget(self.save_viz_button)

        self.export_stats_button = QPushButton("Export Statistics")
        self.export_stats_button.clicked.connect(self._export_statistics)
        self.export_stats_button.setEnabled(False)
        export_layout.addWidget(self.export_stats_button)

        layout.addLayout(export_layout)

        return tab

    def _load_settings(self):
        """Load application settings."""
        settings = QSettings("DatasetGenerator", "LLMFineTuning")

        # Window geometry
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

        # Window state
        state = settings.value("windowState")
        if state:
            self.restoreState(state)

        # Custom validation rules
        rules_json = settings.value("customValidationRules", "[]")
        try:
            self.custom_validation_rules = json.loads(rules_json)
        except json.JSONDecodeError:
            self.custom_validation_rules = []
            logger.error("Failed to load custom validation rules from settings.")


    def _save_settings(self):
        """Save application settings."""
        settings = QSettings("DatasetGenerator", "LLMFineTuning")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        settings.setValue("customValidationRules", json.dumps(self.custom_validation_rules))


    def closeEvent(self, event):
        """Handle window close event."""
        # Save settings
        self._save_settings()
        event.accept()

    def open_dataset(self):
        """Open a dataset file (triggered from menu/toolbar)."""
        # Switch to convert tab and trigger file dialog
        self.tabs.setCurrentIndex(0)
        self._browse_input_file()

    def save_dataset(self):
        """Save the current dataset (triggered from menu/toolbar)."""
        # Determine which tab we're on to know what to save
        current_tab = self.tabs.currentIndex()

        if current_tab == 0:  # Convert tab
            self._convert_dataset()
        elif current_tab == 1:  # Create tab
            self._export_created_dataset()
        elif current_tab == 2:  # Edit tab
            self._save_edited_dataset()
        else:
            QMessageBox.information(self, "Save Dataset", "No active dataset to save from the current tab.")

    def export_dataset(self):
        """Export dataset with format options (triggered from menu/toolbar)."""
        # This will be a generic export, prompting for file and format
        data_to_export = []
        current_tab = self.tabs.currentIndex()

        if current_tab == 0: # Convert tab
            data_to_export = self.output_data
        elif current_tab == 1: # Create tab
            data_to_export = self.output_data
        elif current_tab == 2: # Edit tab
            data_to_export = self.input_data # Edited data is in input_data for this tab
        elif current_tab == 3: # Validate tab
            data_to_export = self.input_data # Data loaded for validation
        elif current_tab == 4: # Visualize tab
            data_to_export = self.input_data # Data loaded for visualization

        if not data_to_export:
            self.show_error("No data available to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Dataset As...", "",
            "JSON Lines (*.jsonl);;JSON (*.json);;CSV (*.csv);;Text (*.txt);;Excel (*.xlsx);;"
            "Parquet (*.parquet);;HDF5 (*.hdf5);;SQLite (*.db);;XML (*.xml);;YAML (*.yaml);;Avro (*.avro);;TSV (*.tsv);;All Files (*)"
        )

        if file_path:
            # Determine format from extension
            ext = os.path.splitext(file_path)[1].lower()[1:]
            if ext.startswith('.'):
                ext = ext[1:]

            format_type = ext
            if ext in ['xlsx', 'xls']:
                format_type = 'excel'
            elif ext == 'db':
                format_type = 'sqlite'
            elif ext not in OUTPUT_FORMATS:
                # Default to jsonl if extension is not recognized
                format_type = 'jsonl'
                QMessageBox.warning(self, "Unknown Format", f"File extension '{ext}' not recognized. Defaulting to JSON Lines format.")

            try:
                self.generator.save_data(data_to_export, file_path, format_type)
                QMessageBox.information(self, "Export Successful", f"Successfully exported {len(data_to_export)} records to {file_path}")
            except Exception as e:
                self.show_error(f"Error exporting data: {str(e)}")

    def validate_dataset_menu(self):
        """Validate the current dataset (triggered from menu/toolbar)."""
        # Switch to validate tab and trigger validation
        self.tabs.setCurrentIndex(3)
        # If data is already loaded in another tab, try to transfer it
        if self.input_data:
            self._update_validate_data_table()
            self.validation_results_text.clear()
            self.validation_results_text.append(f"Loaded {len(self.input_data)} records. Ready for validation.")
            self.status_bar.showMessage(f"Loaded {len(self.input_data)} records")
        else:
            self.show_error("No dataset loaded. Please load a dataset first in the 'Convert' or 'Edit' tab.")


    def fix_dataset_menu(self):
        """Auto-fix issues in the dataset (triggered from menu/toolbar)."""
        # Switch to validate tab and trigger fix
        self.tabs.setCurrentIndex(3)
        self._auto_fix_validation_issues()

    def augment_dataset(self):
        """Augment the dataset with various techniques."""
        # Check if we have data to augment
        data_to_augment = self.output_data if self.output_data else self.input_data
        if not data_to_augment:
            self.show_error("No dataset loaded. Please load or generate a dataset first.")
            return

        # Check available augmentation methods
        available_methods = ["shuffle"]
        if NLTK_AVAILABLE:
            available_methods.append("synonym")
        if FAKER_AVAILABLE:
            available_methods.append("rephrase")

        if not available_methods:
            self.show_error("No augmentation capabilities. Please install 'nltk' and/or 'faker' for augmentation options.")
            return

        # Create dialog for augmentation configuration
        dialog = QDialog(self)
        dialog.setWindowTitle("Augment Dataset")
        dialog.setMinimumWidth(500)

        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Configure data augmentation:"))

        # Augmentation factor
        factor_group = QGroupBox("Augmentation Factor")
        factor_layout = QVBoxLayout(factor_group)

        factor_label = QLabel("Each record will be augmented this many times:")
        factor_layout.addWidget(factor_label)

        factor_slider = QSlider(Qt.Horizontal)
        factor_slider.setRange(1, 10)
        factor_slider.setValue(2)
        factor_slider.setTickPosition(QSlider.TicksBelow)
        factor_slider.setTickInterval(1)
        factor_layout.addWidget(factor_slider)

        factor_value = QLabel("2x (will create 1 additional copy of each record)")
        factor_layout.addWidget(factor_value)

        def update_factor_label(value):
            factor_value.setText(f"{value}x (will create {value-1} additional copies of each record)")
        factor_slider.valueChanged.connect(update_factor_label)
        layout.addWidget(factor_group)

        # Augmentation methods
        methods_group = QGroupBox("Augmentation Methods")
        methods_layout = QVBoxLayout(methods_group)

        method_checkboxes = {}
        for method in available_methods:
            checkbox = QCheckBox(method.capitalize())
            checkbox.setChecked(True)
            methods_layout.addWidget(checkbox)
            method_checkboxes[method] = checkbox

        # Add method descriptions
        methods_layout.addWidget(QLabel("Method descriptions:"))
        descriptions = QTextEdit()
        descriptions.setReadOnly(True)
        descriptions.setMaximumHeight(100)
        descriptions.setText(
            "Shuffle: Randomly shuffle words in text fields (local)\n"
            "Synonym: Replace some words with synonyms (requires NLTK)\n"
            "Rephrase: Generate similar but different text (requires Faker)"
        )
        methods_layout.addWidget(descriptions)

        layout.addWidget(methods_group)

        # Template selection (important for structured augmentation)
        template_group = QGroupBox("Template Selection (for structured augmentation)")
        template_layout = QVBoxLayout(template_group)

        template_label = QLabel("Select a template to guide augmentation (e.g., only augment 'prompt' field):")
        template_layout.addWidget(template_label)

        template_combo = QComboBox()
        template_combo.addItem("None (augment all top-level string fields)")
        for template_name, template_info in DATASET_TEMPLATES.items():
            template_combo.addItem(template_name)
        template_layout.addWidget(template_combo)

        # Pre-select based on current template if available
        if hasattr(self, 'current_template') and self.current_template:
            index = template_combo.findText(self.current_template)
            if index >= 0:
                template_combo.setCurrentIndex(index)

        layout.addWidget(template_group)

        # Output options
        output_group = QGroupBox("Output Options")
        output_layout = QVBoxLayout(output_group)

        output_mode_label = QLabel("How to handle augmented data:")
        output_layout.addWidget(output_mode_label)

        output_mode_combo = QComboBox()
        output_mode_combo.addItem("Add to existing data")
        output_mode_combo.addItem("Replace existing data")
        output_mode_combo.addItem("Save to new file")
        output_layout.addWidget(output_mode_combo)

        # Save path (initially hidden)
        save_layout = QHBoxLayout()
        save_layout.addWidget(QLabel("Save to:"))
        save_path_edit = QLineEdit()
        save_path_edit.setPlaceholderText("Path for augmented data...")
        save_layout.addWidget(save_path_edit)

        save_browse_button = QPushButton("Browse...")
        save_layout.addWidget(save_browse_button)
        output_layout.addLayout(save_layout)

        # Output format (initially hidden)
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Format:"))
        format_combo = QComboBox()
        for format_name in OUTPUT_FORMATS:
            format_combo.addItem(format_name)
        format_combo.setCurrentText("jsonl")
        format_layout.addWidget(format_combo)
        output_layout.addLayout(format_layout)

        # Initially hide save options
        save_path_edit.setVisible(False)
        save_browse_button.setVisible(False)
        format_combo.setVisible(False)

        def update_save_options(mode):
            show_save_options = mode == "Save to new file"
            save_path_edit.setVisible(show_save_options)
            save_browse_button.setVisible(show_save_options)
            format_combo.setVisible(show_save_options)
        output_mode_combo.currentTextChanged.connect(update_save_options)

        def browse_save_path():
            file_path, _ = QFileDialog.getSaveFileName(
                dialog, "Save Augmented Data", "",
                "All Files (*)"
            )
            if file_path:
                save_path_edit.setText(file_path)

                # Auto-detect format from extension
                ext = os.path.splitext(file_path)[1].lower()[1:]
                if ext in OUTPUT_FORMATS:
                    index = format_combo.findText(ext)
                    if index >= 0:
                        format_combo.setCurrentIndex(index)

        save_browse_button.clicked.connect(browse_save_path)

        layout.addWidget(output_group)

        # Add buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("Augment Dataset")
        cancel_button = QPushButton("Cancel")

        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        # Connect buttons
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)

        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            # Get augmentation parameters
            factor = factor_slider.value()
            methods = [m for m, cb in method_checkboxes.items() if cb.isChecked()]
            template_for_augmentation = template_combo.currentText()
            if template_for_augmentation == "None (augment all top-level string fields)":
                template_for_augmentation = None # Pass None to the generator
            output_mode = output_mode_combo.currentText()

            if not methods:
                self.show_error("Please select at least one augmentation method.")
                return

            # Prepare for augmentation
            self.current_operation = "augment"
            self.progress_bar.setVisible(True)
            self.status_bar.showMessage("Augmenting dataset...")

            # Using the worker thread
            self.worker.configure(
                "augment",
                data=data_to_augment,
                factor=factor,
                methods=methods,
                template_name=template_for_augmentation
            )

            # Connect to a special slot for augmentation completion
            self.worker.finished_signal.disconnect()
            self.worker.finished_signal.connect(
                lambda result: self._augment_finished(
                    result, output_mode, save_path_edit.text(), format_combo.currentText()
                )
            )

            self.worker.start()

    def _augment_finished(self, result, output_mode, save_path=None, format_type=None):
        """Handle completion of dataset augmentation."""
        self.progress_bar.setVisible(False)

        if result:
            augmented_data = result
            original_count = len(self.output_data if self.output_data else self.input_data)
            augmented_count = len(augmented_data)
            new_records = augmented_count - original_count

            if output_mode == "Add to existing data":
                if self.output_data:
                    self.output_data = augmented_data
                else:
                    self.input_data = augmented_data
                # Update the table of the current tab if it's visible and relevant
                current_tab_index = self.tabs.currentIndex()
                if current_tab_index == 0: # Convert tab
                    self._update_preview_table()
                elif current_tab_index == 2: # Edit tab
                    self._update_edit_data_table()
                elif current_tab_index == 1: # Create tab
                    self._update_created_data_table()

                QMessageBox.information(
                    self,
                    "Augmentation Complete",
                    f"Added {new_records} augmented records to the dataset."
                )

            elif output_mode == "Replace existing data":
                if self.output_data:
                    self.output_data = augmented_data
                else:
                    self.input_data = augmented_data
                # Update the table of the current tab if it's visible and relevant
                current_tab_index = self.tabs.currentIndex()
                if current_tab_index == 0: # Convert tab
                    self._update_preview_table()
                elif current_tab_index == 2: # Edit tab
                    self._update_edit_data_table()
                elif current_tab_index == 1: # Create tab
                    self._update_created_data_table()

                QMessageBox.information(
                    self,
                    "Augmentation Complete",
                    f"Dataset replaced with {augmented_count} records (includes {new_records} augmented records)."
                )

            elif output_mode == "Save to new file" and save_path:
                # Save to new file
                try:
                    self.generator.save_data(augmented_data, save_path, format_type)
                    QMessageBox.information(
                        self,
                        "Augmentation Complete",
                        f"Saved {augmented_count} records (includes {new_records} augmented records) to {save_path}"
                    )
                except Exception as e:
                    self.show_error(f"Error saving augmented data: {str(e)}")

            self.status_bar.showMessage("Data augmentation complete")
        else:
            self.show_error("Error augmenting dataset.")

        # Reconnect the general finished signal handler
        self.worker.finished_signal.disconnect()
        self.worker.finished_signal.connect(self.process_finished)

    def anonymize_dataset(self):
        """Anonymize sensitive data in the dataset."""
        # Check if we have data to anonymize
        data_to_anonymize = self.output_data if self.output_data else self.input_data
        if not data_to_anonymize:
            self.show_error("No dataset loaded. Please load or generate a dataset first.")
            return

        # Check if faker is available
        if not FAKER_AVAILABLE:
            self.show_error("The 'Faker' library is required for anonymization. Please install it with 'pip install Faker'.")
            return

        # Create dialog for anonymization configuration
        dialog = QDialog(self)
        dialog.setWindowTitle("Anonymize Dataset")
        dialog.setMinimumWidth(500)

        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Select fields to anonymize:"))

        sensitive_patterns = {
            "email": re.compile(r'email|mail|e-mail', re.I),
            "name": re.compile(r'name|username|user|author', re.I),
            "phone": re.compile(r'phone|tel|mobile', re.I),
            "address": re.compile(r'address|location|city|state|zip|postal', re.I),
            "id": re.compile(r'id$|identifier|account', re.I),
            "ip": re.compile(r'ip$|ip_address', re.I),
            "ssn": re.compile(r'ssn|social|security', re.I),
            "credit_card": re.compile(r'credit|card|payment', re.I),
            "dob": re.compile(r'birth|dob|birthday', re.I),
        }

        # Get all field names from the first record
        all_fields = set()
        if data_to_anonymize:
            sample = data_to_anonymize[0]
            for field in sample.keys():
                all_fields.add(field)

        # Create field selection group
        fields_group = QGroupBox("Fields to Anonymize")
        fields_layout = QVBoxLayout(fields_group)

        field_checkboxes = {}
        field_type_combos = {}

        for field in sorted(list(all_fields)):
            field_str = str(field)
            field_lower = field_str.lower()
            detected_type = None

            # Auto-detect field type based on name
            for field_type, pattern in sensitive_patterns.items():
                if pattern.search(field_lower):
                    detected_type = field_type
                    break

            field_layout = QHBoxLayout()
            checkbox = QCheckBox(field_str)
            checkbox.setChecked(detected_type is not None)
            field_layout.addWidget(checkbox)
            field_checkboxes[field_str] = checkbox

            type_combo = QComboBox()
            # Add all Faker providers that generate simple string/number data
            faker_providers = [
                "address", "am_pm", "ascii_email", "ascii_free_email", "ascii_safe_email",
                "bank_country", "bban", "binary", "boolean", "bothify", "bs", "building_number",
                "catch_phrase", "century", "chrome", "city", "city_prefix", "city_suffix",
                "color_name", "company", "company_email", "company_suffix", "continent",
                "country", "country_code", "credit_card_expire", "credit_card_full",
                "credit_card_number", "credit_card_provider", "credit_card_security_code",
                "currency_code", "currency_name", "currency_symbol", "date", "date_of_birth",
                "date_object", "date_time", "date_time_ad", "date_time_between",
                "date_time_this_century", "date_time_this_decade", "date_time_this_month",
                "date_time_this_year", "day_of_month", "day_of_week", "dga", "domain_name",
                "domain_word", "ean", "ean13", "ean8", "email", "emoji", "file_extension",
                "file_name", "file_path", "firefox", "first_name", "first_name_female",
                "first_name_male", "fixed_width", "free_email", "free_email_domain",
                "future_date", "future_datetime", "hex_color", "hostname", "http_method",
                "iban", "image_url", "internet_explorer", "ipv4", "ipv4_network_class",
                "ipv6", "isbn10", "isbn13", "iso8601", "job", "language_code", "last_name",
                "last_name_female", "last_name_male", "latitude", "lexify", "license_plate",
                "locale", "localized_datetime", "longitude", "mac_address", "mac_platform_token",
                "md5", "mime_type", "month", "month_name", "msisdn", "name", "name_female",
                "name_male", "nic_handle", "numerify", "opera", "paragraph", "paragraphs",
                "password", "past_date", "past_datetime", "phone_number", "postcode",
                "prefix", "prefix_female", "prefix_male", "profile", "pybool", "pyfloat",
                "pyint", "pystr", "pystr_format", "pylist", "pydict", "pyset", "pytuple",
                "random_digit", "random_digit_not_null", "random_element", "random_int",
                "random_letter", "random_number", "random_sample", "random_uppercase_letter",
                "safari", "sentence", "sentences", "sha1", "sha256", "ssn", "state", "state_abbr",
                "street_address", "street_name", "street_suffix", "suffix", "suffix_female",
                "suffix_male", "text", "time", "time_delta", "time_object", "time_series",
                "timezone", "tld", "uri", "uri_extension", "uri_page", "uri_path", "url",
                "user_agent", "user_name", "uuid4", "word", "words", "year", "zipcode",
                "zipcode_plus4"
            ]
            for provider in sorted(faker_providers):
                type_combo.addItem(provider.replace('_', ' ').title())

            if detected_type:
                index = type_combo.findText(detected_type.replace('_', ' ').title())
                if index >= 0:
                    type_combo.setCurrentIndex(index)

            field_layout.addWidget(type_combo)
            field_type_combos[field_str] = type_combo
            fields_layout.addLayout(field_layout)

        fields_layout.addStretch()
        layout.addWidget(fields_group)

        # Output options
        output_group = QGroupBox("Output Options")
        output_layout = QVBoxLayout(output_group)

        output_mode_label = QLabel("How to handle anonymized data:")
        output_layout.addWidget(output_mode_label)

        output_mode_combo = QComboBox()
        output_mode_combo.addItem("Replace existing data")
        output_mode_combo.addItem("Save to new file")
        output_layout.addWidget(output_mode_combo)

        # Save path (initially hidden)
        save_layout = QHBoxLayout()
        save_layout.addWidget(QLabel("Save to:"))
        save_path_edit = QLineEdit()
        save_path_edit.setPlaceholderText("Path for anonymized data...")
        save_layout.addWidget(save_path_edit)

        save_browse_button = QPushButton("Browse...")
        save_layout.addWidget(save_browse_button)
        output_layout.addLayout(save_layout)

        # Output format (initially hidden)
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Format:"))
        format_combo = QComboBox()
        for format_name in OUTPUT_FORMATS:
            format_combo.addItem(format_name)
        format_combo.setCurrentText("jsonl")
        format_layout.addWidget(format_combo)
        output_layout.addLayout(format_layout)

        # Initially hide save options
        save_path_edit.setVisible(False)
        save_browse_button.setVisible(False)
        format_combo.setVisible(False)

        def update_save_options(mode):
            show_save_options = mode == "Save to new file"
            save_path_edit.setVisible(show_save_options)
            save_browse_button.setVisible(show_save_options)
            format_combo.setVisible(show_save_options)
        output_mode_combo.currentTextChanged.connect(update_save_options)

        def browse_save_path():
            file_path, _ = QFileDialog.getSaveFileName(
                dialog, "Save Anonymized Data", "",
                "All Files (*)"
            )
            if file_path:
                save_path_edit.setText(file_path)

                # Auto-detect format from extension
                ext = os.path.splitext(file_path)[1].lower()[1:]
                if ext in OUTPUT_FORMATS:
                    index = format_combo.findText(ext)
                    if index >= 0:
                        format_combo.setCurrentIndex(index)

        save_browse_button.clicked.connect(browse_save_path)

        layout.addWidget(output_group)

        # Add buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("Anonymize Dataset")
        cancel_button = QPushButton("Cancel")

        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        # Connect buttons
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)

        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            # Get selected fields and their types
            fields_to_anonymize = {}

            for field_str, checkbox in field_checkboxes.items():
                if checkbox.isChecked():
                    # Convert 'Title Case' back to 'snake_case' for Faker
                    faker_method_name = field_type_combos[field_str].currentText().replace(' ', '_').lower()
                    fields_to_anonymize[field_str] = faker_method_name

            if not fields_to_anonymize:
                self.show_error("Please select at least one field to anonymize.")
                return

            # Prepare for anonymization
            self.current_operation = "anonymize"
            self.progress_bar.setVisible(True)
            self.status_bar.showMessage("Anonymizing dataset...")

            # Using the worker thread
            self.worker.configure(
                "anonymize",
                data=data_to_anonymize,
                fields=fields_to_anonymize
            )

            # Connect to a special slot for anonymization completion
            self.worker.finished_signal.disconnect()
            self.worker.finished_signal.connect(
                lambda result: self._anonymize_finished(
                    result, output_mode_combo.currentText(), save_path_edit.text(), format_combo.currentText()
                )
            )

            self.worker.start()

    def _anonymize_finished(self, result, output_mode, save_path=None, format_type=None):
        """Handle completion of dataset anonymization."""
        self.progress_bar.setVisible(False)

        if result:
            anonymized_data = result

            if output_mode == "Replace existing data":
                if self.output_data:
                    self.output_data = anonymized_data
                else:
                    self.input_data = anonymized_data
                # Update the table of the current tab if it's visible and relevant
                current_tab_index = self.tabs.currentIndex()
                if current_tab_index == 0: # Convert tab
                    self._update_preview_table()
                elif current_tab_index == 2: # Edit tab
                    self._update_edit_data_table()
                elif current_tab_index == 1: # Create tab
                    self._update_created_data_table()

                QMessageBox.information(
                    self,
                    "Anonymization Complete",
                    f"Successfully anonymized sensitive data in the dataset."
                )

            elif output_mode == "Save to new file" and save_path:
                # Save to new file
                try:
                    self.generator.save_data(anonymized_data, save_path, format_type)
                    QMessageBox.information(
                        self,
                        "Anonymization Complete",
                        f"Saved anonymized dataset to {save_path}"
                    )
                except Exception as e:
                    self.show_error(f"Error saving anonymized data: {str(e)}")

            self.status_bar.showMessage("Data anonymization complete")
        else:
            self.show_error("Error anonymizing dataset.")

        # Reconnect the general finished signal handler
        self.worker.finished_signal.disconnect()
        self.worker.finished_signal.connect(self.process_finished)

    def split_dataset(self):
        """Split the dataset into train, validation, and test sets."""
        data_to_split = self.output_data if self.output_data else self.input_data
        if not data_to_split:
            self.show_error("No dataset loaded. Please load or generate a dataset first.")
            return

        if not SKLEARN_AVAILABLE:
            self.show_error("The 'scikit-learn' library is required for dataset splitting. Please install it with 'pip install scikit-learn'.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Split Dataset")
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout(dialog)

        layout.addWidget(QLabel("Configure dataset splitting ratios:"))

        form_layout = QFormLayout()
        self.train_ratio_spin = QDoubleSpinBox()
        self.train_ratio_spin.setRange(0.0, 1.0)
        self.train_ratio_spin.setSingleStep(0.05)
        self.train_ratio_spin.setValue(0.8)
        form_layout.addRow("Train Ratio:", self.train_ratio_spin)

        self.valid_ratio_spin = QDoubleSpinBox()
        self.valid_ratio_spin.setRange(0.0, 1.0)
        self.valid_ratio_spin.setSingleStep(0.05)
        self.valid_ratio_spin.setValue(0.1)
        form_layout.addRow("Validation Ratio:", self.valid_ratio_spin)

        self.test_ratio_spin = QDoubleSpinBox()
        self.test_ratio_spin.setRange(0.0, 1.0)
        self.test_ratio_spin.setSingleStep(0.05)
        self.test_ratio_spin.setValue(0.1)
        form_layout.addRow("Test Ratio:", self.test_ratio_spin)

        self.random_seed_split_check = QCheckBox("Use random seed for reproducibility")
        self.random_seed_split_check.setChecked(True)
        form_layout.addRow("Random Seed:", self.random_seed_split_check)

        self.random_seed_split_spin = QSpinBox()
        self.random_seed_split_spin.setRange(0, 999999)
        self.random_seed_split_spin.setValue(42)
        self.random_seed_split_spin.setEnabled(True)
        self.random_seed_split_check.toggled.connect(self.random_seed_split_spin.setEnabled)
        form_layout.addRow("Seed Value:", self.random_seed_split_spin)


        layout.addLayout(form_layout)

        output_group = QGroupBox("Output Options")
        output_layout = QVBoxLayout(output_group)
        output_layout.addWidget(QLabel("Save split datasets to:"))

        self.train_path_edit = QLineEdit()
        self.train_path_edit.setPlaceholderText("train.jsonl")
        train_browse_button = QPushButton("Browse...")
        train_browse_button.clicked.connect(lambda: self._browse_save_path(self.train_path_edit, "train.jsonl"))
        train_h_layout = QHBoxLayout()
        train_h_layout.addWidget(self.train_path_edit)
        train_h_layout.addWidget(train_browse_button)
        output_layout.addLayout(train_h_layout)

        self.valid_path_edit = QLineEdit()
        self.valid_path_edit.setPlaceholderText("validation.jsonl (optional)")
        valid_browse_button = QPushButton("Browse...")
        valid_browse_button.clicked.connect(lambda: self._browse_save_path(self.valid_path_edit, "validation.jsonl"))
        valid_h_layout = QHBoxLayout()
        valid_h_layout.addWidget(self.valid_path_edit)
        valid_h_layout.addWidget(valid_browse_button)
        output_layout.addLayout(valid_h_layout)

        self.test_path_edit = QLineEdit()
        self.test_path_edit.setPlaceholderText("test.jsonl (optional)")
        test_browse_button = QPushButton("Browse...")
        test_browse_button.clicked.connect(lambda: self._browse_save_path(self.test_path_edit, "test.jsonl"))
        test_h_layout = QHBoxLayout()
        test_h_layout.addWidget(self.test_path_edit)
        test_h_layout.addWidget(test_browse_button)
        output_layout.addLayout(test_h_layout)

        self.split_output_format_combo = QComboBox()
        for format_name, description in OUTPUT_FORMATS.items():
            self.split_output_format_combo.addItemWithDescription(format_name, description)
        self.split_output_format_combo.setCurrentText("jsonl")
        output_layout.addWidget(QLabel("Output Format for Splits:"))
        output_layout.addWidget(self.split_output_format_combo)

        layout.addWidget(output_group)

        button_box = QHBoxLayout()
        ok_button = QPushButton("Split Dataset")
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        button_box.addWidget(ok_button)
        button_box.addWidget(cancel_button)
        layout.addLayout(button_box)

        if dialog.exec_() == QDialog.Accepted:
            train_ratio = self.train_ratio_spin.value()
            valid_ratio = self.valid_ratio_spin.value()
            test_ratio = self.test_ratio_spin.value()
            random_seed = self.random_seed_split_spin.value() if self.random_seed_split_check.isChecked() else None
            output_format = self.split_output_format_combo.currentText()

            total_ratio = train_ratio + valid_ratio + test_ratio
            if not (0.99 <= total_ratio <= 1.01):
                self.show_error("Sum of ratios must be approximately 1.0. Please adjust.")
                return

            train_path = self.train_path_edit.text()
            valid_path = self.valid_path_edit.text()
            test_path = self.test_path_edit.text()

            if not train_path:
                self.show_error("Train set output path is required.")
                return

            self.current_operation = "split"
            self.progress_bar.setVisible(True)
            self.status_bar.showMessage("Splitting dataset...")

            self.worker.configure(
                "split",
                data=data_to_split,
                train_ratio=train_ratio,
                valid_ratio=valid_ratio,
                test_ratio=test_ratio,
                random_seed=random_seed,
                output_format=output_format,
                train_path=train_path,
                valid_path=valid_path,
                test_path=test_path
            )
            self.worker.finished_signal.disconnect()
            self.worker.finished_signal.connect(self._split_finished)
            self.worker.start()

    def _split_finished(self, result):
        """Handle completion of dataset splitting."""
        self.progress_bar.setVisible(False)
        if result:
            QMessageBox.information(self, "Split Complete", "Dataset successfully split and saved.")
            self.status_bar.showMessage("Dataset splitting complete.")
        else:
            self.show_error("Error splitting dataset.")
        self.worker.finished_signal.disconnect()
        self.worker.finished_signal.connect(self.process_finished) # Reconnect general handler

    def _browse_save_path(self, line_edit: QLineEdit, default_name: str):
        """Helper to browse for a save file path."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save File", default_name,
            "All Files (*)"
        )
        if file_path:
            line_edit.setText(file_path)


    def visualize_dataset_menu(self):
        """Visualize the current dataset (triggered from menu/toolbar)."""
        # Switch to visualize tab and trigger visualization
        self.tabs.setCurrentIndex(4)
        # If data is already loaded in another tab, try to transfer it
        if self.input_data:
            self._update_visualize_data_table()
            self._update_stats_display()
            self.status_bar.showMessage(f"Loaded {len(self.input_data)} records for visualization.")
        else:
            self.show_error("No dataset loaded. Please load a dataset first in the 'Convert' or 'Edit' tab.")


    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Dataset Generator",
            "<h1>Dataset Generator for LLM Fine-Tuning (Offline)</h1>"
            "<p>Version 2.0</p>"
            "<p>A comprehensive toolkit for creating, editing, and processing datasets "
            "specifically designed for fine-tuning language models.</p>"
            "<p>This version operates entirely offline, with no external API dependencies.</p>"
            "<p>Developed by Your Name/Organization</p>"
        )

    def show_help(self):
        """Show help information."""
        QMessageBox.information(
            self,
            "Help",
            "<h2>Dataset Generator Help</h2>"
            "<p><b>Convert Tab:</b> Import data, map fields to a template, and export to various formats.</p>"
            "<p><b>Create Tab:</b> Generate new datasets from scratch using templates and local content generation.</p>"
            "<p><b>Edit Tab:</b> Load, add, delete, filter, and directly edit records in an existing dataset.</p>"
            "<p><b>Validate Tab:</b> Check datasets for common issues (missing values, type inconsistencies, duplicates) "
            "and apply custom validation rules. Auto-fix functionality available.</p>"
            "<p><b>Visualize Tab:</b> Generate graphical summaries and statistics of your dataset.</p>"
            "<p><b>Tools Menu:</b> Access advanced operations like Augmentation, Anonymization, and Splitting.</p>"
            "<p><b>Drag & Drop:</b> Drag and drop files onto the preview tables to load them.</p>"
        )

    def update_status(self, message):
        """Update status bar message."""
        self.status_bar.showMessage(message)

    def update_progress_bar(self, current, total):
        """Update progress bar."""
        if total > 0:
            self.progress_bar.setValue(int((current / total) * 100))
            self.progress_bar.setVisible(True)
        else:
            self.progress_bar.setVisible(False)

    def show_error(self, message):
        """Show error message."""
        QMessageBox.critical(self, "Error", message)
        self.status_bar.showMessage("Error occurred")
        self.progress_bar.setVisible(False)


    def process_finished(self, result):
        """Handle completion of background process."""
        self.progress_bar.setVisible(False)
        if hasattr(self, 'current_operation'):
            if self.current_operation == "load":
                self.input_data = result
                self._update_preview_table()
                self.map_fields_button.setEnabled(True)
                self.status_bar.showMessage(f"Loaded {len(result)} records")

            elif self.current_operation == "process":
                self.output_data = result
                self._update_preview_table()
                self.preview_button.setEnabled(True)
                self.convert_button.setEnabled(True)
                self.status_bar.showMessage(f"Processed {len(result)} records")

            elif self.current_operation == "save":
                if result:
                    QMessageBox.information(self, "Save Successful", "Dataset successfully saved.")
                    self.status_bar.showMessage("Dataset saved.")
                else:
                    self.show_error("Failed to save dataset.")

            elif self.current_operation == "generate":
                self.output_data = result
                self._update_created_data_table()
                QMessageBox.information(self, "Generation Complete", f"Generated {len(result)} records.")
                self.status_bar.showMessage(f"Generated {len(result)} records.")
            # Add more result handling for other operations as needed
            del self.current_operation # Clear current operation

    def _on_tab_changed(self, index):
        """Handle tab change event."""
        # Reset status bar and progress bar when changing tabs
        self.status_bar.showMessage("Ready")
        self.progress_bar.setVisible(False)

        # Update data tables based on current tab's data
        if index == 0: # Convert tab
            self._update_preview_table()
        elif index == 1: # Create tab
            self._update_created_data_table()
        elif index == 2: # Edit tab
            self._update_edit_data_table()
        elif index == 3: # Validate tab
            self._update_validate_data_table()
        elif index == 4: # Visualize tab
            self._update_visualize_data_table()
            self._update_stats_display()


    def _handle_dropped_file(self, file_path: str):
        """Handle file dropped onto the preview table."""
        self.input_path_edit.setText(file_path)
        self._load_dataset()

    def _browse_input_file(self):
        """Browse for input file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Input File", "",
            "All Supported Files (" +
            " ".join([f"*.{ext}" for ext in INPUT_FORMATS.keys() if ext not in ['excel', 'sqlite']]) +
            " *.xlsx *.xls *.db" +
            ");;All Files (*)"
        )

        if file_path:
            self.input_path_edit.setText(file_path)

            # Auto-detect format from extension
            ext = os.path.splitext(file_path)[1].lower()[1:]
            if ext.startswith('.'):
                ext = ext[1:]

            if ext in ['xlsx', 'xls']:
                ext = 'excel'
            elif ext == 'db':
                ext = 'sqlite'

            if ext in INPUT_FORMATS:
                index = self.input_format_combo.findText(ext)
                if index >= 0:
                    self.input_format_combo.setCurrentIndex(index)

    def _browse_output_file(self):
        """Browse for output file location."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select Output File", "",
            "All Files (*)"
        )

        if file_path:
            self.output_path_edit.setText(file_path)

    def _browse_create_output_file(self):
        """Browse for create tab output file location."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select Output File", "",
            "All Files (*)"
        )

        if file_path:
            self.create_output_path_edit.setText(file_path)

    def _browse_edit_file(self):
        """Browse for edit tab input file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select File to Edit", "",
            "All Supported Files (" +
            " ".join([f"*.{ext}" for ext in INPUT_FORMATS.keys() if ext not in ['excel', 'sqlite']]) +
            " *.xlsx *.xls *.db" +
            ");;All Files (*)"
        )

        if file_path:
            self.edit_path_edit.setText(file_path)

            # Auto-detect format from extension
            ext = os.path.splitext(file_path)[1].lower()[1:]
            if ext.startswith('.'):
                ext = ext[1:]

            if ext in ['xlsx', 'xls']:
                ext = 'excel'
            elif ext == 'db':
                ext = 'sqlite'

            if ext in INPUT_FORMATS:
                index = self.edit_format_combo.findText(ext)
                if index >= 0:
                    self.edit_format_combo.setCurrentIndex(index)

    def _browse_edit_output_file(self):
        """Browse for edit tab output file location."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select Output File", "",
            "All Files (*)"
        )

        if file_path:
            self.edit_output_path_edit.setText(file_path)

    def _browse_validate_file(self):
        """Browse for validate tab input file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select File to Validate", "",
            "All Supported Files (" +
            " ".join([f"*.{ext}" for ext in INPUT_FORMATS.keys() if ext not in ['excel', 'sqlite']]) +
            " *.xlsx *.xls *.db" +
            ");;All Files (*)"
        )

        if file_path:
            self.validate_path_edit.setText(file_path)

            # Auto-detect format from extension
            ext = os.path.splitext(file_path)[1].lower()[1:]
            if ext.startswith('.'):
                ext = ext[1:]

            if ext in ['xlsx', 'xls']:
                ext = 'excel'
            elif ext == 'db':
                ext = 'sqlite'

            if ext in INPUT_FORMATS:
                index = self.validate_format_combo.findText(ext)
                if index >= 0:
                    self.validate_format_combo.setCurrentIndex(index)

    def _browse_visualize_file(self):
        """Browse for visualize tab input file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select File to Visualize", "",
            "All Supported Files (" +
            " ".join([f"*.{ext}" for ext in INPUT_FORMATS.keys() if ext not in ['excel', 'sqlite']]) +
            " *.xlsx *.xls *.db" +
            ");;All Files (*)"
        )

        if file_path:
            self.visualize_path_edit.setText(file_path)

            # Auto-detect format from extension
            ext = os.path.splitext(file_path)[1].lower()[1:]
            if ext.startswith('.'):
                ext = ext[1:]

            if ext in ['xlsx', 'xls']:
                ext = 'excel'
            elif ext == 'db':
                ext = 'sqlite'

            if ext in INPUT_FORMATS:
                index = self.visualize_format_combo.findText(ext)
                if index >= 0:
                    self.visualize_format_combo.setCurrentIndex(index)

    def _load_dataset(self):
        """Load dataset from the specified file (Convert tab)."""
        file_path = self.input_path_edit.text()
        format_type = self.input_format_combo.currentText()

        if not file_path:
            self.show_error("Please select an input file.")
            return

        if not os.path.exists(file_path):
            self.show_error(f"File not found: {file_path}")
            return

        self.current_operation = "load"
        self.progress_bar.setVisible(True)
        self.status_bar.showMessage(f"Loading data from {file_path}...")

        self.worker.configure(
            "load",
            file_path=file_path,
            format_type=format_type
        )
        self.worker.start()

    def _load_edit_dataset(self):
        """Load dataset for editing (Edit tab)."""
        file_path = self.edit_path_edit.text()
        format_type = self.edit_format_combo.currentText()

        if not file_path:
            self.show_error("Please select an input file.")
            return

        if not os.path.exists(file_path):
            self.show_error(f"File not found: {file_path}")
            return

        try:
            self.generator.load_data(file_path, format_type)
            self.input_data = self.generator.input_data # Data for editing is stored in input_data
            self._update_edit_data_table()
            self.status_bar.showMessage(f"Loaded {len(self.input_data)} records for editing.")
        except Exception as e:
            self.show_error(f"Error loading data: {str(e)}")

    def _load_validate_dataset(self):
        """Load dataset for validation (Validate tab)."""
        file_path = self.validate_path_edit.text()
        format_type = self.validate_format_combo.currentText()

        if not file_path:
            self.show_error("Please select an input file.")
            return

        if not os.path.exists(file_path):
            self.show_error(f"File not found: {file_path}")
            return

        try:
            self.generator.load_data(file_path, format_type)
            self.input_data = self.generator.input_data # Data for validation is stored in input_data
            self._update_validate_data_table()
            self.validation_results_text.clear()
            self.validation_results_text.append(f"Loaded {len(self.input_data)} records. Ready for validation.")
            self.status_bar.showMessage(f"Loaded {len(self.input_data)} records for validation.")
        except Exception as e:
            self.show_error(f"Error loading data: {str(e)}")

    def _load_visualize_dataset(self):
        """Load dataset for visualization (Visualize tab)."""
        file_path = self.visualize_path_edit.text()
        format_type = self.visualize_format_combo.currentText()

        if not file_path:
            self.show_error("Please select an input file.")
            return

        if not os.path.exists(file_path):
            self.show_error(f"File not found: {file_path}")
            return

        try:
            self.generator.load_data(file_path, format_type)
            self.input_data = self.generator.input_data # Data for visualization is stored in input_data
            self._update_visualize_data_table()
            self._update_stats_display() # Update statistics display
            self.status_bar.showMessage(f"Loaded {len(self.input_data)} records for visualization.")
        except Exception as e:
            self.show_error(f"Error loading data: {str(e)}")

    def _update_template_info(self, template_name):
        """Update template information display."""
        if template_name in DATASET_TEMPLATES:
            info = DATASET_TEMPLATES[template_name]
            self.template_info_label.setText(info["description"])

            # Show structure example
            structure_str = json.dumps(info["structure"], indent=2)
            self.template_structure_text.setText(structure_str)

    def _update_create_template_info(self, template_name):
        """Update template information in the create tab."""
        if template_name in DATASET_TEMPLATES:
            info = DATASET_TEMPLATES[template_name]
            self.create_template_info_label.setText(info["description"])

    def _show_field_mapping_dialog(self):
        """Show dialog for mapping input fields to template fields."""
        if not self.input_data:
            self.show_error("No input data loaded. Please load a dataset first in the 'Convert' tab.")
            return

        template_name = self.template_combo.currentText()
        if template_name not in DATASET_TEMPLATES:
            self.show_error("Please select a valid template.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Map Fields")
        dialog.setMinimumWidth(600)
        dialog.setModal(True)

        main_layout = QVBoxLayout(dialog)
        main_layout.addWidget(QLabel(f"Map input fields to '{template_name}' template fields:"))

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        content_widget = QWidget()
        form_layout = QFormLayout(content_widget)
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)

        # Get template structure
        template_structure = DATASET_TEMPLATES[template_name]["structure"]
        mapping_widgets = {}

        # Get all unique input fields from the loaded data
        all_input_fields = set()
        if self.input_data:
            for record in self.input_data:
                # Recursively extract all keys, including nested ones (flattened with dot notation)
                def extract_keys(obj, path=""):
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            current_path = f"{path}.{k}" if path else k
                            all_input_fields.add(current_path)
                            extract_keys(v, current_path)
                    elif isinstance(obj, list):
                        for i, item in enumerate(obj):
                            # For lists, we don't add index to the field name for mapping purposes
                            # as the mapping is about the *type* of item in the list
                            extract_keys(item, path) # Pass current path without index

                extract_keys(self.input_data[0]) # Use first record to get available fields

        sorted_input_fields = sorted(list(all_input_fields))
        input_field_model = QStringListModel(["Skip"] + sorted_input_fields)

        # Helper function to process structure recursively
        def process_structure(structure, prefix="", current_form_layout=form_layout):
            for key, value in structure.items():
                full_key = f"{prefix}{key}"
                if isinstance(value, dict):
                    group = QGroupBox(key)
                    group_layout = QFormLayout(group)
                    current_form_layout.addRow(group)
                    process_structure(value, f"{full_key}.", group_layout)
                elif isinstance(value, list) and value and isinstance(value[0], dict):
                    group = QGroupBox(f"{key} (List of Objects)")
                    group_layout = QFormLayout(group)
                    current_form_layout.addRow(group)
                    # For lists of objects, we map the fields within the object structure
                    # We assume all items in the list have the same structure as the first template item
                    process_structure(value[0], f"{full_key}[*].", group_layout) # Use [*] to denote list items
                else:
                    combo = QComboBox()
                    combo.setModel(input_field_model)
                    combo.setEditable(True) # Allow user to type in custom field names
                    completer = QCompleter(input_field_model, combo)
                    combo.setCompleter(completer)

                    # Try to pre-select based on exact match or common patterns
                    default_index = 0 # 'Skip'
                    if full_key in sorted_input_fields:
                        default_index = sorted_input_fields.index(full_key) + 1
                    elif key in sorted_input_fields: # Try just the last part of the key
                        default_index = sorted_input_fields.index(key) + 1
                    combo.setCurrentIndex(default_index)

                    current_form_layout.addRow(QLabel(f"Template '{full_key}':"), combo)
                    mapping_widgets[full_key] = combo

        # Process template structure
        process_structure(template_structure)

        # Add buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")

        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        main_layout.addLayout(button_layout)

        # Connect buttons
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)

        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            # Collect mapping
            mapping = {}
            for key, combo in mapping_widgets.items():
                input_field = combo.currentText().strip()
                if input_field and input_field.lower() != "skip":
                    mapping[key] = input_field

            self.field_mapping = mapping
            self.current_template = template_name

            self.status_bar.showMessage(f"Field mapping set with {len(mapping)} mappings.")
            self.preview_button.setEnabled(True)
            self.convert_button.setEnabled(True) # Enable convert button after mapping

    def _preview_converted_data(self):
        """Preview the data after conversion."""
        if not self.input_data:
            self.show_error("No input data loaded. Please load a dataset first.")
            return

        if not self.field_mapping:
            self.show_error("Please map fields first.")
            return

        if not self.current_template:
            self.show_error("Please select a template first.")
            return

        # Set up and start worker thread
        self.current_operation = "process"
        self.progress_bar.setVisible(True)
        self.status_bar.showMessage("Processing data for preview...")

        self.worker.configure(
            "process",
            template=self.current_template,
            mapping=self.field_mapping,
            input_data=self.input_data
        )
        self.worker.start()

    def _convert_dataset(self):
        """Convert and save the dataset."""
        if not self.output_data:
            # If output_data is empty, try to process first
            try:
                self._preview_converted_data()
                # Wait for processing to finish, then save.
                # This is a synchronous call for simplicity; in a real app,
                # you'd chain signals or use a QFutureWatcher.
                # For now, rely on _preview_converted_data's worker.
                # The save will be triggered by process_finished if current_operation is "process"
                # and then "save" is set.
                self.worker.finished_signal.disconnect() # Disconnect the default handler
                self.worker.finished_signal.connect(self._chain_process_to_save)
                return
            except Exception as e:
                self.show_error(f"Error preparing data for conversion: {str(e)}")
                return

        output_path = self.output_path_edit.text()
        format_type = self.output_format_combo.currentText()

        if not output_path:
            self.show_error("Please specify an output file path.")
            return

        self.current_operation = "save"
        self.progress_bar.setVisible(True)
        self.status_bar.showMessage(f"Saving data to {output_path}...")

        self.worker.configure(
            "save",
            output_path=output_path,
            format_type=format_type,
            output_data=self.output_data
        )
        self.worker.start()

    def _chain_process_to_save(self, processed_data):
        """Helper to chain processing completion to saving."""
        self.output_data = processed_data
        self.worker.finished_signal.disconnect() # Disconnect this temporary handler
        self.worker.finished_signal.connect(self.process_finished) # Reconnect default

        if processed_data:
            self._convert_dataset() # Now actually save
        else:
            self.show_error("Processing failed, cannot save.")

    def _generate_dataset(self):
        """Generate a new dataset from scratch."""
        template_name = self.create_template_combo.currentText()
        num_records = self.num_records_spin.value()
        auto_generate = self.auto_generate_check.isChecked()
        use_random_seed = self.random_seed_check.isChecked()
        random_seed_value = self.random_seed_spin.value()

        if template_name not in DATASET_TEMPLATES:
            self.show_error("Please select a valid template.")
            return

        template_structure = DATASET_TEMPLATES[template_name]["structure"]

        self.current_operation = "generate"
        self.progress_bar.setVisible(True)
        self.status_bar.showMessage(f"Generating {num_records} records...")

        # Configure worker for generation
        self.worker.configure(
            "generate",
            template_structure=template_structure,
            num_records=num_records,
            auto_generate=auto_generate,
            use_random_seed=use_random_seed,
            random_seed_value=random_seed
        )
        self.worker.start()

    def _export_created_dataset(self):
        """Export the generated dataset."""
        if not self.output_data:
            self.show_error("No data to export. Please generate data first.")
            return

        output_path = self.create_output_path_edit.text()
        format_type = self.create_format_combo.currentText()

        if not output_path:
            self.show_error("Please specify an output file path.")
            return

        try:
            self.generator.save_data(self.output_data, output_path, format_type)
            self.status_bar.showMessage(f"Saved {len(self.output_data)} records to {output_path}")
            QMessageBox.information(self, "Export Successful", f"Successfully exported {len(self.output_data)} records to {output_path}")
        except Exception as e:
            self.show_error(f"Error exporting data: {str(e)}")

    def _save_edited_dataset(self):
        """Save the edited dataset."""
        if not self.input_data:  # Edited data is stored in input_data for the edit tab
            self.show_error("No data to save. Please load a dataset first.")
            return

        output_path = self.edit_output_path_edit.text()
        format_type = self.edit_output_format_combo.currentText()

        if not output_path:
            self.show_error("Please specify an output file path.")
            return

        try:
            self.generator.save_data(self.input_data, output_path, format_type)
            self.status_bar.showMessage(f"Saved {len(self.input_data)} records to {output_path}")
            QMessageBox.information(self, "Save Successful", f"Successfully saved {len(self.input_data)} records to {output_path}")
        except Exception as e:
            self.show_error(f"Error saving data: {str(e)}")

    def _add_record(self):
        """Add a new record to the dataset."""
        if not self.input_data:
            # If no data loaded, create a basic record based on a default template
            template_structure = DATASET_TEMPLATES["instruction"]["structure"]
            new_record = {}
            for key, value in template_structure.items():
                new_record[key] = "" # Empty values
            self.input_data.append(new_record)
            self._update_edit_data_table()
            self.status_bar.showMessage("Added a new empty record.")
            return

        # Get fields from the first record to define structure for new record
        sample = self.input_data[0]
        new_record = {}
        for field in sample.keys():
            new_record[field] = "" # Initialize with empty strings

        # Create dialog for adding a record
        dialog = QDialog(self)
        dialog.setWindowTitle("Add New Record")
        dialog.setMinimumWidth(400)
        layout = QFormLayout(dialog)

        field_widgets = {}
        for field, value in new_record.items():
            line_edit = QLineEdit(str(value))
            layout.addRow(f"{field}:", line_edit)
            field_widgets[field] = line_edit

        button_box = QHBoxLayout()
        ok_button = QPushButton("Add Record")
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        button_box.addWidget(ok_button)
        button_box.addWidget(cancel_button)
        layout.addRow(button_box)

        if dialog.exec_() == QDialog.Accepted:
            for field, widget in field_widgets.items():
                new_record[field] = widget.text()
            self.input_data.append(new_record)
            self._update_edit_data_table()
            self.status_bar.showMessage(f"Added new record. Total: {len(self.input_data)}")


    def _handle_edit_table_item_changed(self, item: QTableWidgetItem):
        """Handle direct editing of cells in the edit_data_table."""
        row = item.row()
        col = item.column()
        new_value = item.text()

        # Get the field name from the header
        field_name = self.edit_data_table.horizontalHeaderItem(col).text()

        if 0 <= row < len(self.input_data) and field_name in self.input_data[row]:
            try:
                # Attempt to convert the value back to its original type if possible
                old_value = self.input_data[row][field_name]
                if isinstance(old_value, int):
                    converted_value = int(new_value)
                elif isinstance(old_value, float):
                    converted_value = float(new_value)
                elif isinstance(old_value, bool):
                    converted_value = new_value.lower() in ['true', '1', 'yes']
                elif isinstance(old_value, (dict, list)):
                    converted_value = json.loads(new_value) # Assume JSON string for complex types
                else:
                    converted_value = new_value # Keep as string

                self.input_data[row][field_name] = converted_value
                self.status_bar.showMessage(f"Cell edited: Record {row+1}, Field '{field_name}' updated.")
            except ValueError:
                self.show_error(f"Invalid input for field '{field_name}'. Please enter a valid value.")
                # Revert to original value in table if conversion fails
                item.setText(str(old_value))
            except json.JSONDecodeError:
                self.show_error(f"Invalid JSON for field '{field_name}'. Please enter valid JSON string.")
                item.setText(str(old_value))
        else:
            self.show_error("Error updating cell: Field not found or invalid row.")


    def _delete_selected_records(self):
        """Delete selected records."""
        selected_rows = sorted(list(set(index.row() for index in self.edit_data_table.selectedIndexes())), reverse=True)

        if not selected_rows:
            self.show_error("Please select records to delete.")
            return

        confirm = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete {len(selected_rows)} records?",
            QMessageBox.Yes | QMessageBox.No
        )

        if confirm == QMessageBox.Yes:
            for row_idx in selected_rows:
                if 0 <= row_idx < len(self.input_data):
                    del self.input_data[row_idx]

            self._update_edit_data_table()
            self.status_bar.showMessage(f"Deleted {len(selected_rows)} records. Remaining: {len(self.input_data)}")

    def _filter_edit_table(self):
        """Filter records in the edit_data_table based on search input."""
        search_text = self.edit_search_input.text().lower()

        for row in range(self.edit_data_table.rowCount()):
            match = False
            for col in range(self.edit_data_table.columnCount()):
                item = self.edit_data_table.item(row, col)
                if item and search_text in item.text().lower():
                    match = True
                    break
            self.edit_data_table.setRowHidden(row, not match)
        self.status_bar.showMessage(f"Filtered records for '{search_text}'.")


    def _edit_custom_validation_rules(self):
        """Open a dialog to edit custom validation rules."""
        # Get all fields from current loaded data for suggestions
        all_fields = []
        if self.input_data:
            # Recursively extract all keys, including nested ones (flattened with dot notation)
            def extract_keys(obj, path=""):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        current_path = f"{path}.{k}" if path else k
                        all_fields.append(current_path)
                        extract_keys(v, current_path)
                elif isinstance(obj, list):
                    for item in obj:
                        extract_keys(item, path) # Pass current path without index
            extract_keys(self.input_data[0]) # Use first record to get available fields
            all_fields = sorted(list(set(all_fields))) # Remove duplicates and sort

        dialog = CustomValidationRuleDialog(self.custom_validation_rules, all_fields, self)
        if dialog.exec_() == QDialog.Accepted:
            self.custom_validation_rules = dialog.get_rules()
            QMessageBox.information(self, "Rules Saved", f"Saved {len(self.custom_validation_rules)} custom validation rules.")
            logger.info(f"Custom validation rules updated: {self.custom_validation_rules}")

    def _validate_dataset_action(self):
        """Validate the dataset in the validate tab."""
        if not self.input_data:
            self.show_error("No dataset loaded. Please load a dataset first.")
            return

        template_name = self.validate_template_combo.currentText()
        
        # Clear previous results
        self.validation_results_text.clear()
        
        all_problems = []

        # 1. Basic template validation (missing fields, structural issues)
        try:
            template_problems = self.generator.detect_and_fix_problems(self.input_data.copy(), template_name)
            if template_problems:
                all_problems.extend([f"Template/Structure Issue: {p}" for p in template_problems])
            else:
                self.validation_results_text.append("Template structure check: PASSED.")
        except Exception as e:
            all_problems.append(f"Error during template validation: {str(e)}")

        # 2. Additional built-in checks
        if self.check_missing_values.isChecked():
            missing_report = self._validate_missing_values()
            if missing_report != "No missing values found.":
                all_problems.append("\n--- Missing Values Check ---")
                all_problems.append(missing_report)
            else:
                self.validation_results_text.append("Missing values check: PASSED.")

        if self.check_field_types.isChecked():
            type_report = self._validate_field_types()
            if type_report != "No field type inconsistencies found.":
                all_problems.append("\n--- Field Type Consistency Check ---")
                all_problems.append(type_report)
            else:
                self.validation_results_text.append("Field type consistency check: PASSED.")

        if self.check_duplicate_records.isChecked():
            duplicates_report = self._validate_duplicates()
            if duplicates_report != "No duplicate records found.":
                all_problems.append("\n--- Duplicate Records Check ---")
                all_problems.append(duplicates_report)
            else:
                self.validation_results_text.append("Duplicate records check: PASSED.")

        # 3. Custom validation rules
        if self.custom_validation_rules:
            custom_rule_problems = self._validate_custom_rules()
            if custom_rule_problems:
                all_problems.append("\n--- Custom Validation Rules Check ---")
                all_problems.extend(custom_rule_problems)
            else:
                self.validation_results_text.append("Custom rules check: PASSED.")

        # Display all collected problems
        if all_problems:
            self.validation_results_text.append("\n--- Summary of Issues Found ---")
            for problem in all_problems:
                self.validation_results_text.append(f"- {problem}")
            self.validation_results_text.append(f"\nTotal issues found: {len(all_problems)}")
            self.fix_issues_button.setEnabled(True)
            self.validation_results_text.setStyleSheet("color: red;")
        else:
            self.validation_results_text.append("\n[bold green]Dataset passed all selected validation checks![/bold green]")
            self.validation_results_text.setStyleSheet("color: green;")
            self.fix_issues_button.setEnabled(False)

        self.status_bar.showMessage("Validation complete.")

    def _validate_missing_values(self):
        """Check for missing values in the dataset."""
        if not self.input_data:
            return "No data to validate."

        all_fields = set()
        for item in self.input_data:
            all_fields.update(item.keys())

        missing_counts = {}
        for field in all_fields:
            missing = sum(1 for item in self.input_data if field not in item or item[field] is None or (isinstance(item[field], str) and not item[field].strip()))
            if missing > 0:
                missing_counts[field] = missing

        if missing_counts:
            report = "Issues found:\n"
            for field, count in missing_counts.items():
                report += f"- Field '{field}' has {count} missing/empty values.\n"
        else:
            report = "No missing values found."
        return report

    def _validate_field_types(self):
        """Check for field type consistency."""
        if not self.input_data:
            return "No data to validate."

        # Determine expected types from the first record (or a sample)
        expected_types = {}
        if self.input_data:
            sample_record = self.input_data[0]
            for field, value in sample_record.items():
                if isinstance(value, str):
                    expected_types[field] = "string"
                elif isinstance(value, (int, float)):
                    expected_types[field] = "number"
                elif isinstance(value, bool):
                    expected_types[field] = "boolean"
                elif isinstance(value, list):
                    expected_types[field] = "array"
                elif isinstance(value, dict):
                    expected_types[field] = "object"
                else:
                    expected_types[field] = "unknown"

        type_issues = defaultdict(list)
        for i, item in enumerate(self.input_data):
            for field, expected_type in expected_types.items():
                if field in item and item[field] is not None:
                    value = item[field]
                    actual_type = None
                    if isinstance(value, str):
                        actual_type = "string"
                    elif isinstance(value, (int, float)):
                        actual_type = "number"
                    elif isinstance(value, bool):
                        actual_type = "boolean"
                    elif isinstance(value, list):
                        actual_type = "array"
                    elif isinstance(value, dict):
                        actual_type = "object"

                    if actual_type and actual_type != expected_type:
                        type_issues[field].append(f"Record #{i+1}: Expected '{expected_type}', got '{actual_type}' (value: {value})")

        if type_issues:
            report = "Type inconsistencies found:\n"
            for field, issues in type_issues.items():
                report += f"- Field '{field}' has {len(issues)} inconsistencies. Expected: '{expected_types.get(field, 'N/A')}'\n"
                for issue_detail in issues[:3]: # Show first 3 examples
                    report += f"  - {issue_detail}\n"
                if len(issues) > 3:
                    report += f"  ... ({len(issues) - 3} more issues for this field)\n"
        else:
            report = "No field type inconsistencies found."
        return report

    def _validate_duplicates(self):
        """Check for duplicate records in the dataset."""
        if not self.input_data:
            return "No data to validate."

        record_hashes = {}
        duplicates = []

        for i, item in enumerate(self.input_data):
            try:
                item_str = json.dumps(item, sort_keys=True) # Stable representation
            except TypeError: # Handle non-serializable objects if any
                item_str = str(item) # Fallback to string representation

            if item_str in record_hashes:
                duplicates.append((i, record_hashes[item_str]))
            else:
                record_hashes[item_str] = i

        if duplicates:
            report = f"Found {len(duplicates)} duplicate records:\n"
            for i, original in duplicates[:5]: # Limit to 5 examples
                report += f"- Record #{i+1} is a duplicate of record #{original+1}\n"
            if len(duplicates) > 5:
                report += f"  (and {len(duplicates) - 5} more...)\n"
        else:
            report = "No duplicate records found."
        return report

    def _validate_custom_rules(self) -> List[str]:
        """Apply custom validation rules to the dataset."""
        problems = []
        if not self.custom_validation_rules:
            return problems # No custom rules defined

        for i, record in enumerate(self.input_data):
            for rule in self.custom_validation_rules:
                field_name = rule.get("field")
                rule_type = rule.get("type")
                rule_value = rule.get("value")

                # Determine which fields to apply the rule to
                fields_to_check = []
                if field_name == "ALL_FIELDS":
                    fields_to_check = list(record.keys())
                elif field_name in record:
                    fields_to_check = [field_name]
                else:
                    # If field_name is a nested path, try to find it
                    nested_value = get_nested_value(record, field_name, default=None)
                    if nested_value is not None:
                        fields_to_check = [field_name] # Treat as a single field for rule application
                    else:
                        # If field doesn't exist, and it's a 'Required' rule, it's a problem
                        if rule_type == "Required":
                            problems.append(f"Record #{i+1}: Missing required field '{field_name}'.")
                        continue # Skip other checks if field doesn't exist

                for current_field in fields_to_check:
                    # Get the value, handling nested paths
                    value = get_nested_value(record, current_field)

                    is_valid = True
                    error_message = ""

                    if rule_type == "Required":
                        if value is None or (isinstance(value, str) and not value.strip()):
                            is_valid = False
                            error_message = f"Field '{current_field}' is required but is missing or empty."
                    elif rule_type == "Type Check":
                        expected_type_str = str(rule_value).lower()
                        if expected_type_str == "string":
                            is_valid = isinstance(value, str)
                        elif expected_type_str == "number":
                            is_valid = isinstance(value, (int, float))
                        elif expected_type_str == "boolean":
                            is_valid = isinstance(value, bool)
                        elif expected_type_str == "list":
                            is_valid = isinstance(value, list)
                        elif expected_type_str == "dict":
                            is_valid = isinstance(value, dict)
                        if not is_valid:
                            error_message = f"Field '{current_field}' type mismatch: Expected '{expected_type_str}', got '{type(value).__name__}'."
                    elif rule_type == "Regex Match":
                        try:
                            if not isinstance(value, str) or not re.match(str(rule_value), value):
                                is_valid = False
                                error_message = f"Field '{current_field}' does not match regex pattern '{rule_value}'."
                        except re.error:
                            error_message = f"Invalid regex pattern: '{rule_value}' for field '{current_field}'."
                            is_valid = False # Treat invalid regex as an error
                    elif rule_type == "Min Length":
                        try:
                            min_len = int(rule_value)
                            if not isinstance(value, str) or len(value) < min_len:
                                is_valid = False
                                error_message = f"Field '{current_field}' length ({len(str(value))}) is less than min length {min_len}."
                        except ValueError:
                            error_message = f"Invalid min length value: '{rule_value}' for field '{current_field}'."
                            is_valid = False
                    elif rule_type == "Max Length":
                        try:
                            max_len = int(rule_value)
                            if not isinstance(value, str) or len(value) > max_len:
                                is_valid = False
                                error_message = f"Field '{current_field}' length ({len(str(value))}) is greater than max length {max_len}."
                        except ValueError:
                            error_message = f"Invalid max length value: '{rule_value}' for field '{current_field}'."
                            is_valid = False
                    elif rule_type == "Min Value":
                        try:
                            min_val = float(rule_value)
                            if not isinstance(value, (int, float)) or value < min_val:
                                is_valid = False
                                error_message = f"Field '{current_field}' value ({value}) is less than min value {min_val}."
                        except ValueError:
                            error_message = f"Invalid min value: '{rule_value}' for field '{current_field}'."
                            is_valid = False
                    elif rule_type == "Max Value":
                        try:
                            max_val = float(rule_value)
                            if not isinstance(value, (int, float)) or value > max_val:
                                is_valid = False
                                error_message = f"Field '{current_field}' value ({value}) is greater than max value {max_val}."
                        except ValueError:
                            error_message = f"Invalid max value: '{rule_value}' for field '{current_field}'."
                            is_valid = False
                    elif rule_type == "Custom Python":
                        try:
                            # Evaluate the lambda function
                            func = eval(rule_value)
                            if not func(value):
                                is_valid = False
                                error_message = f"Field '{current_field}' failed custom rule: '{rule_value}'."
                        except Exception as e:
                            error_message = f"Error evaluating custom Python rule for field '{current_field}': {e}"
                            is_valid = False

                    if not is_valid:
                        problems.append(f"Record #{i+1}, Field '{current_field}': {error_message}")
        return problems

    def _auto_fix_validation_issues(self):
        """Auto-fix validation issues."""
        if not self.input_data:
            self.show_error("No dataset loaded. Please load a dataset first.")
            return

        template_name = self.validate_template_combo.currentText()

        self.current_operation = "fix"
        self.progress_bar.setVisible(True)
        self.status_bar.showMessage("Auto-fixing dataset issues...")

        # Use a copy of the data for the worker to avoid direct modification issues
        data_copy = [item.copy() for item in self.input_data]

        self.worker.configure(
            "fix",
            data=data_copy,
            template_name=template_name
        )
        # Connect to a special slot for fix completion
        self.worker.finished_signal.disconnect()
        self.worker.finished_signal.connect(self._fix_finished)
        self.worker.start()

    def _fix_finished(self, fixed_data):
        """Handle completion of auto-fix operation."""
        self.progress_bar.setVisible(False)
        if fixed_data:
            self.input_data = fixed_data # Update the main data with fixed data
            self._update_validate_data_table()
            QMessageBox.information(self, "Auto-Fix Complete", "Dataset issues have been automatically fixed.")
            self.status_bar.showMessage("Auto-fix complete.")
            self._validate_dataset_action() # Re-run validation to show new status
        else:
            self.show_error("Error during auto-fix operation.")

        # Reconnect the general finished signal handler
        self.worker.finished_signal.disconnect()
        self.worker.finished_signal.connect(self.process_finished)

    def _generate_visualizations(self):
        """Generate dataset visualizations."""
        if not self.input_data:
            self.show_error("No dataset loaded. Please load a dataset first.")
            return

        if not VISUALIZATION_AVAILABLE:
            self.show_error("Visualization requires 'matplotlib' and 'seaborn' libraries. Please install them with 'pip install matplotlib seaborn'.")
            return

        selected_field_for_dist = self.viz_field_combo.currentText() if self.viz_field_combo.count() > 0 else None

        self.current_operation = "visualize"
        self.progress_bar.setVisible(True)
        self.status_bar.showMessage("Generating visualizations...")

        # Create a temporary file for the visualization
        temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name

        self.worker.configure(
            "visualize",
            data=self.input_data,
            output_path=temp_file_path,
            selected_field=selected_field_for_dist
        )
        self.worker.finished_signal.disconnect()
        self.worker.finished_signal.connect(lambda success: self._visualize_finished(success, temp_file_path))
        self.worker.start()

    def _visualize_finished(self, success, temp_file_path):
        """Handle completion of visualization operation."""
        self.progress_bar.setVisible(False)
        if success:
            try:
                pixmap = QPixmap(temp_file_path)
                if not pixmap.isNull():
                    viz_width = self.viz_placeholder.width()
                    viz_height = self.viz_placeholder.height()
                    if pixmap.width() > viz_width or pixmap.height() > viz_height:
                        pixmap = pixmap.scaled(viz_width, viz_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.viz_placeholder.setPixmap(pixmap)
                    self.viz_placeholder.setText("") # Clear text if image loaded
                else:
                    self.viz_placeholder.setText("Failed to load visualization image.")
                    self.show_error("Failed to load visualization image.")

                self._update_stats_display() # Refresh stats display
                self.save_viz_button.setEnabled(True)
                self.export_stats_button.setEnabled(True)
                self.status_bar.showMessage("Visualization complete.")

            except Exception as e:
                self.show_error(f"Error displaying visualization: {str(e)}")
        else:
            self.show_error("Error generating visualizations.")

        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        self.worker.finished_signal.disconnect()
        self.worker.finished_signal.connect(self.process_finished) # Reconnect general handler


    def _save_visualizations(self):
        """Save visualizations to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Visualization", "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )

        if file_path:
            pixmap = self.viz_placeholder.pixmap()
            if pixmap and not pixmap.isNull():
                pixmap.save(file_path)
                self.status_bar.showMessage(f"Visualization saved to {file_path}")
            else:
                self.show_error("No visualization to save.")

    def _export_statistics(self):
        """Export statistics to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Statistics", "",
            "Text Files (*.txt);;JSON Files (*.json);;All Files (*)"
        )

        if file_path:
            stats_text = self.stats_text.toPlainText()
            ext = os.path.splitext(file_path)[1].lower()

            try:
                if ext == '.json':
                    # Convert statistics dict to JSON
                    stats_dict = self.generator.statistics # Assuming generator holds the latest stats
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(stats_dict, f, indent=2, ensure_ascii=False)
                else: # Default to text file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(stats_text)
                self.status_bar.showMessage(f"Statistics exported to {file_path}")
            except Exception as e:
                self.show_error(f"Error exporting statistics: {str(e)}")

    def _update_preview_table(self):
        """Update the preview table (Convert tab) with current data."""
        data = self.output_data if self.output_data else self.input_data

        self.preview_table.clearContents()
        self.preview_table.setRowCount(0)
        self.preview_table.setColumnCount(0)

        if not data:
            self.preview_table.setHorizontalHeaderLabels([])
            return

        # Get all field names
        all_fields = set()
        for item in data:
            all_fields.update(item.keys())
        sorted_fields = sorted(list(all_fields))

        # Set up table
        self.preview_table.setRowCount(min(len(data), 100))  # Limit to 100 rows for performance
        self.preview_table.setColumnCount(len(sorted_fields))

        # Set headers
        self.preview_table.setHorizontalHeaderLabels(sorted_fields)

        # Fill data
        for row_idx, item in enumerate(data[:100]):
            for col_idx, field in enumerate(sorted_fields):
                value = item.get(field, "")
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False, indent=2)
                table_item = QTableWidgetItem(str(value))
                self.preview_table.setItem(row_idx, col_idx, table_item)
        self.preview_table.resizeColumnsToContents()

    def _update_created_data_table(self):
        """Update the created data table (Create tab)."""
        data = self.output_data

        self.created_data_table.clearContents()
        self.created_data_table.setRowCount(0)
        self.created_data_table.setColumnCount(0)

        if not data:
            self.created_data_table.setHorizontalHeaderLabels([])
            return

        all_fields = set()
        for item in data:
            all_fields.update(item.keys())
        sorted_fields = sorted(list(all_fields))

        self.created_data_table.setRowCount(min(len(data), 100))
        self.created_data_table.setColumnCount(len(sorted_fields))
        self.created_data_table.setHorizontalHeaderLabels(sorted_fields)

        for row_idx, item in enumerate(data[:100]):
            for col_idx, field in enumerate(sorted_fields):
                value = item.get(field, "")
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False, indent=2)
                table_item = QTableWidgetItem(str(value))
                self.created_data_table.setItem(row_idx, col_idx, table_item)
        self.created_data_table.resizeColumnsToContents()

    def _update_edit_data_table(self):
        """Update the edit data table (Edit tab)."""
        data = self.input_data # Data for editing is stored in input_data

        self.edit_data_table.clearContents()
        self.edit_data_table.setRowCount(0)
        self.edit_data_table.setColumnCount(0)

        if not data:
            self.edit_data_table.setHorizontalHeaderLabels([])
            return

        all_fields = set()
        for item in data:
            all_fields.update(item.keys())
        sorted_fields = sorted(list(all_fields))

        self.edit_data_table.setRowCount(len(data)) # Show all rows for editing
        self.edit_data_table.setColumnCount(len(sorted_fields))
        self.edit_data_table.setHorizontalHeaderLabels(sorted_fields)

        for row_idx, item in enumerate(data):
            for col_idx, field in enumerate(sorted_fields):
                value = item.get(field, "")
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False, indent=2)
                table_item = QTableWidgetItem(str(value))
                self.edit_data_table.setItem(row_idx, col_idx, table_item)
        self.edit_data_table.resizeColumnsToContents()
        self._filter_edit_table() # Apply current filter after updating

    def _update_validate_data_table(self):
        """Update the validate data table (Validate tab)."""
        data = self.input_data

        self.validate_data_table.clearContents()
        self.validate_data_table.setRowCount(0)
        self.validate_data_table.setColumnCount(0)

        if not data:
            self.validate_data_table.setHorizontalHeaderLabels([])
            return

        all_fields = set()
        for item in data:
            all_fields.update(item.keys())
        sorted_fields = sorted(list(all_fields))

        self.validate_data_table.setRowCount(min(len(data), 100))
        self.validate_data_table.setColumnCount(len(sorted_fields))
        self.validate_data_table.setHorizontalHeaderLabels(sorted_fields)

        for row_idx, item in enumerate(data[:100]):
            for col_idx, field in enumerate(sorted_fields):
                value = item.get(field, "")
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False, indent=2)
                table_item = QTableWidgetItem(str(value))
                self.validate_data_table.setItem(row_idx, col_idx, table_item)
        self.validate_data_table.resizeColumnsToContents()

    def _update_visualize_data_table(self):
        """Update the visualize data table (Visualize tab)."""
        data = self.input_data

        # This table is just for display, not direct editing.
        # Reusing validate_data_table logic for simplicity.
        self.validate_data_table.clearContents()
        self.validate_data_table.setRowCount(0)
        self.validate_data_table.setColumnCount(0)

        if not data:
            self.visualize_data_table.setHorizontalHeaderLabels([])
            return

        all_fields = set()
        for item in data:
            all_fields.update(item.keys())
        sorted_fields = sorted(list(all_fields))

        self.visualize_data_table.setRowCount(min(len(data), 100))
        self.visualize_data_table.setColumnCount(len(sorted_fields))
        self.visualize_data_table.setHorizontalHeaderLabels(sorted_fields)

        for row_idx, item in enumerate(data[:100]):
            for col_idx, field in enumerate(sorted_fields):
                value = item.get(field, "")
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False, indent=2)
                table_item = QTableWidgetItem(str(value))
                self.visualize_data_table.setItem(row_idx, col_idx, table_item)
        self.visualize_data_table.resizeColumnsToContents()

        # Populate field selector for visualization
        self.viz_field_combo.clear()
        if self.input_data:
            sample = self.input_data[0]
            for field in sorted(sample.keys()):
                self.viz_field_combo.addItem(field)

    def _update_stats_display(self):
        """Update the statistics display in the Visualize tab."""
        self.stats_text.clear()
        if not self.input_data:
            self.stats_text.setText("No data loaded to display statistics.")
            return

        self.generator._calculate_statistics() # Recalculate based on current input_data
        stats = self.generator.statistics

        self.stats_text.append("<b>Dataset Statistics:</b>")
        self.stats_text.append(f"- Total Records: {stats.get('record_count', 0)}")
        self.stats_text.append(f"- Number of Fields: {len(stats.get('field_types', {}))}")

        self.stats_text.append("\n<b>Field Types:</b>")
        if stats.get('field_types'):
            for field, f_type in stats['field_types'].items():
                self.stats_text.append(f"  - <i>{field}</i>: {f_type}")
        else:
            self.stats_text.append("  No field type information available.")

        self.stats_text.append("\n<b>Missing Values:</b>")
        if stats.get('missing_values'):
            has_missing = False
            for field, count in stats['missing_values'].items():
                if count > 0:
                    self.stats_text.append(f"  - <i>{field}</i>: {count} missing")
                    has_missing = True
            if not has_missing:
                self.stats_text.append("  No missing values detected.")
        else:
            self.stats_text.append("  No missing values detected.")

        self.stats_text.append("\n<b>Common Values (Top 5 for String Fields):</b>")
        if stats.get('common_values'):
            has_common = False
            for field, common_list in stats['common_values'].items():
                if common_list:
                    self.stats_text.append(f"  - <i>{field}</i>:")
                    for value, count in common_list:
                        self.stats_text.append(f"    - '{value}': {count} times")
                    has_common = True
            if not has_common:
                self.stats_text.append("  No common values for string fields.")
        else:
            self.stats_text.append("  No common values for string fields.")


class ProcessingWorker(QThread):
    """Worker thread for dataset processing operations."""
    progress_signal = pyqtSignal(int, int)  # current, total
    status_signal = pyqtSignal(str)  # status message
    error_signal = pyqtSignal(str)  # error message
    finished_signal = pyqtSignal(object)  # Result object

    def __init__(self, parent=None):
        super().__init__(parent)
        self.generator = DatasetGenerator()
        self.operation = None
        self.params = {}

    def configure(self, operation, **params):
        """Configure the worker with an operation and parameters."""
        self.operation = operation
        self.params = params

    def run(self):
        """Main execution method."""
        try:
            self.status_signal.emit(f"Starting {self.operation} operation...")

            if self.operation == "load":
                self.status_signal.emit(f"Loading data from {self.params['file_path']}...")
                self.generator.load_data(
                    self.params['file_path'],
                    self.params.get('format_type'),
                    self.params.get('table_name')
                )
                self.finished_signal.emit(self.generator.input_data)

            elif self.operation == "process":
                self.status_signal.emit("Processing data...")
                # The generator's process_data needs input_data, template, mapping
                processed_data = self.generator.process_data(
                    input_data=self.params['input_data'],
                    template_name=self.params['template'],
                    mapping=self.params['mapping']
                )
                self.finished_signal.emit(processed_data)

            elif self.operation == "save":
                self.status_signal.emit(f"Saving data to {self.params['output_path']}...")
                self.generator.save_data(
                    self.params['output_data'],
                    self.params['output_path'],
                    self.params['format_type']
                )
                self.finished_signal.emit(True)

            elif self.operation == "generate":
                self.status_signal.emit("Generating new dataset...")
                num_records = self.params['num_records']
                template_structure = self.params['template_structure']
                auto_generate = self.params['auto_generate']
                use_random_seed = self.params['use_random_seed']
                random_seed_value = self.params['random_seed_value']

                if use_random_seed:
                    random.seed(random_seed_value)

                generated_data = []
                for i in range(num_records):
                    self.progress_signal.emit(i, num_records)
                    record = {}

                    def fill_structure(structure, current_record):
                        for key, value in structure.items():
                            if isinstance(value, dict):
                                current_record[key] = {}
                                fill_structure(value, current_record[key])
                            elif isinstance(value, list) and value and isinstance(value[0], dict):
                                current_record[key] = []
                                for _ in range(random.randint(1, 3)): # Generate 1-3 items for lists
                                    list_item = {}
                                    fill_structure(value[0], list_item)
                                    current_record[key].append(list_item)
                            else:
                                # Generate content based on auto_generate and field name hints
                                if auto_generate and FAKER_AVAILABLE:
                                    if "name" in key.lower():
                                        current_record[key] = faker.name()
                                    elif "email" in key.lower():
                                        current_record[key] = faker.email()
                                    elif "phone" in key.lower():
                                        current_record[key] = faker.phone_number()
                                    elif "address" in key.lower():
                                        current_record[key] = faker.address()
                                    elif "company" in key.lower():
                                        current_record[key] = faker.company()
                                    elif "content" in key.lower() or "text" in key.lower() or "prompt" in key.lower() or "completion" in key.lower() or "answer" in key.lower():
                                        current_record[key] = faker.paragraph(nb_sentences=random.randint(1, 5))
                                    elif "question" in key.lower():
                                        current_record[key] = faker.sentence(nb_words=random.randint(5, 15)) + "?"
                                    elif "id" in key.lower():
                                        current_record[key] = str(uuid.uuid4())
                                    elif "number" in key.lower():
                                        current_record[key] = random.randint(1, 1000)
                                    elif "bool" in key.lower():
                                        current_record[key] = random.choice([True, False])
                                    else:
                                        current_record[key] = faker.word() + " " + str(random.randint(1, 100)) # Default generic text
                                else:
                                    # Fallback to simple random content if Faker not available or not auto-generating
                                    if isinstance(value, str):
                                        current_record[key] = f"Sample_{key}_{random.randint(1, 100)}"
                                    elif isinstance(value, int):
                                        current_record[key] = random.randint(1, 100)
                                    elif isinstance(value, bool):
                                        current_record[key] = random.choice([True, False])
                                    else:
                                        current_record[key] = "" # Default empty

                    fill_structure(template_structure, record)
                    generated_data.append(record)

                self.progress_signal.emit(num_records, num_records) # Ensure 100%
                self.finished_signal.emit(generated_data)

            elif self.operation == "validate_custom":
                self.status_signal.emit("Validating dataset with custom rules...")
                data = self.params['data']
                custom_rules = self.params['custom_rules']
                # This part needs to be handled by the main thread or a dedicated validator class
                # as it involves eval(), which is generally unsafe in workers.
                # For now, I'll return a placeholder and remind myself to refactor if eval is critical.
                # The _validate_custom_rules in MainWindow already handles this directly.
                self.finished_signal.emit({"result": "Custom validation handled in main thread."})

            elif self.operation == "split":
                self.status_signal.emit("Splitting dataset...")
                train, valid, test = self.generator.split_dataset(
                    self.params['data'],
                    self.params.get('train_ratio', 0.8),
                    self.params.get('valid_ratio', 0.1),
                    self.params.get('test_ratio', 0.1),
                    self.params.get('random_seed')
                )
                output_format = self.params['output_format']
                train_path = self.params['train_path']
                valid_path = self.params['valid_path']
                test_path = self.params['test_path']

                self.generator.save_data(train, train_path, output_format)
                if valid:
                    self.generator.save_data(valid, valid_path, output_format)
                if test:
                    self.generator.save_data(test, test_path, output_format)

                self.finished_signal.emit(True) # Indicate success

            elif self.operation == "fix":
                self.status_signal.emit("Auto-fixing dataset issues...")
                # The data passed to fix is already a copy
                fixed_data = self.generator.detect_and_fix_problems(
                    self.params['data'],
                    self.params.get('template_name')
                )
                self.finished_signal.emit(self.params['data']) # Return the modified data copy

            elif self.operation == "augment":
                self.status_signal.emit("Augmenting dataset...")
                # The data passed to augment is already a copy
                augmented_data = self.generator.augment_data(
                    self.params['data'],
                    self.params.get('factor', 2),
                    self.params.get('methods'),
                    self.params.get('template_name')
                )
                self.finished_signal.emit(augmented_data)

            elif self.operation == "anonymize":
                self.status_signal.emit("Anonymizing sensitive data...")
                # The data passed to anonymize is already a copy
                anonymized_data = self.generator.anonymize_data(
                    self.params['data'],
                    self.params.get('fields')
                )
                self.finished_signal.emit(anonymized_data)

            elif self.operation == "visualize":
                self.status_signal.emit("Generating visualizations...")
                self.generator.visualize_data(
                    self.params['data'],
                    self.params.get('output_path'),
                    self.params.get('selected_field')
                )
                self.finished_signal.emit(True) # Indicate success

            else:
                self.error_signal.emit(f"Unknown operation: {self.operation}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_signal.emit(str(e))


# --- CLI Mode (for direct script execution) ---
def cli_mode():
    """Parse command line arguments and run in CLI mode."""
    parser = argparse.ArgumentParser(description="Dataset Generator for LLM Fine-Tuning")

    parser.add_argument('-i', '--input', help='Input file path')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-if', '--input-format', choices=INPUT_FORMATS.keys(), help='Input file format')
    parser.add_argument('-of', '--output-format', choices=OUTPUT_FORMATS.keys(), help='Output file format')
    parser.add_argument('-t', '--template', choices=DATASET_TEMPLATES.keys(), help='Dataset template')
    parser.add_argument('-m', '--mapping', help='Field mapping in JSON format (e.g., \'{"prompt": "text_field"}\')')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive GUI mode')
    parser.add_argument('--validate', action='store_true', help='Validate the dataset after processing')
    parser.add_argument('--fix', action='store_true', help='Auto-fix common issues in the dataset after processing')
    parser.add_argument('--augment', type=int, nargs='?', const=2, help='Augment data. Specify factor (default 2).')
    parser.add_argument('--anonymize', action='store_true', help='Anonymize sensitive fields in the dataset.')
    parser.add_argument('--split', type=float, nargs=3, metavar=('TRAIN', 'VALID', 'TEST'), help='Split dataset (e.g., --split 0.8 0.1 0.1)')
    parser.add_argument('--split-output-paths', nargs=3, metavar=('TRAIN_PATH', 'VALID_PATH', 'TEST_PATH'), help='Output paths for train, validation, test splits.')
    parser.add_argument('--visualize', help='Generate and save visualizations to a specified output image path.')
    parser.add_argument('--num-workers', type=int, default=None, help='Number of worker threads for processing.')
    parser.add_argument('--create-records', type=int, help='Number of records to create from scratch (requires --template).')
    parser.add_argument('--create-output', help='Output path for created dataset.')


    args = parser.parse_args()

    if args.interactive or len(sys.argv) == 1:
        app = QApplication(sys.argv)
        main_window = MainWindow()
        main_window.show()
        sys.exit(app.exec_())
        return

    generator = DatasetGenerator()

    # --- Data Loading ---
    if args.input:
        input_format = args.input_format or os.path.splitext(args.input)[1].lower()[1:]
        if input_format.startswith('.'): input_format = input_format[1:]
        try:
            generator.load_data(args.input, input_format)
        except Exception as e:
            console.print(f"[bold red]Error loading input data: {str(e)}[/bold red]")
            sys.exit(1)
    elif args.create_records:
        if not args.template:
            console.print("[bold red]Error: --create-records requires --template.[/bold red]")
            sys.exit(1)
        console.print(f"Creating {args.create_records} records with template '{args.template}'...")
        template_structure = DATASET_TEMPLATES[args.template]["structure"]
        generated_data = []
        for i in tqdm(range(args.create_records), desc="Generating records"):
            record = {}
            def fill_structure_cli(structure, current_record):
                for key, value in structure.items():
                    if isinstance(value, dict):
                        current_record[key] = {}
                        fill_structure_cli(value, current_record[key])
                    elif isinstance(value, list) and value and isinstance(value[0], dict):
                        current_record[key] = []
                        for _ in range(random.randint(1, 2)): # Simple lists
                            list_item = {}
                            fill_structure_cli(value[0], list_item)
                            current_record[key].append(list_item)
                    else:
                        # Simple placeholder generation for CLI
                        current_record[key] = f"Generated {key} {random.randint(1, 100)}"
            fill_structure_cli(template_structure, record)
            generated_data.append(record)
        generator.input_data = generated_data # Treat generated data as input for further processing
        console.print(f"[bold green]Generated {len(generator.input_data)} records.[/bold green]")
    else:
        console.print("[bold red]Error: Either --input or --create-records must be specified.[/bold red]")
        sys.exit(1)

    # --- Template and Mapping (if converting) ---
    if not args.create_records: # Only apply template/mapping if loading existing data
        if args.template:
            try:
                generator.set_template(args.template)
            except ValueError as e:
                console.print(f"[bold red]Error setting template: {str(e)}[/bold red]")
                sys.exit(1)
        else:
            generator.set_template("instruction") # Default template

        if args.mapping:
            try:
                mapping = json.loads(args.mapping)
                generator.set_field_mapping(mapping)
            except json.JSONDecodeError:
                console.print("[bold red]Error: Invalid JSON mapping format.[/bold red]")
                sys.exit(1)
        else:
            # Attempt a simple 1:1 mapping if no mapping provided and template matches input fields
            if generator.input_data:
                sample_record = generator.input_data[0]
                default_mapping = {}
                template_structure = DATASET_TEMPLATES[generator.template]["structure"]
                def build_default_mapping(struct, prefix=""):
                    for k, v in struct.items():
                        full_key = f"{prefix}{k}"
                        if isinstance(v, dict):
                            build_default_mapping(v, f"{full_key}.")
                        elif isinstance(v, list) and v and isinstance(v[0], dict):
                            build_default_mapping(v[0], f"{full_key}[*].")
                        else:
                            # If template field name exists in input, map it
                            if k in sample_record:
                                default_mapping[full_key] = k
                            elif full_key in sample_record:
                                default_mapping[full_key] = full_key
                build_default_mapping(template_structure)
                generator.set_field_mapping(default_mapping)
                if not default_mapping:
                    console.print("[bold yellow]Warning: No mapping provided and automatic mapping failed. Data might not be processed correctly.[/bold yellow]")


    # --- Processing ---
    if not args.create_records: # If data was loaded, process it
        try:
            generator.process_data(
                input_data=generator.input_data,
                template_name=generator.template,
                mapping=generator.mapping,
                num_workers=args.num_workers
            )
        except Exception as e:
            console.print(f"[bold red]Error processing data: {str(e)}[/bold red]")
            sys.exit(1)
    else: # If data was created, it's already in output_data
        generator.output_data = generator.input_data # Set output_data to generated data for subsequent steps

    # --- Post-processing operations ---
    if args.fix:
        try:
            problems_found = generator.detect_and_fix_problems(generator.output_data, generator.template)
            if problems_found:
                console.print(f"[bold yellow]Fixed {len(problems_found)} issues.[/bold yellow]")
                for p in problems_found[:5]:
                    console.print(f"  - {p}")
                if len(problems_found) > 5:
                    console.print(f"  ...and {len(problems_found) - 5} more.")
            else:
                console.print("[bold green]No issues found or fixed.[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error during auto-fix: {str(e)}[/bold red]")

    if args.augment:
        try:
            generator.augment_data(generator.output_data, augmentation_factor=args.augment, template_name=generator.template)
            console.print(f"[bold green]Dataset augmented by factor {args.augment}. New size: {len(generator.output_data)} records.[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error during augmentation: {str(e)}[/bold red]")

    if args.anonymize:
        try:
            generator.anonymize_data(generator.output_data)
            console.print("[bold green]Sensitive data anonymized.[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error during anonymization: {str(e)}[/bold red]")

    if args.validate:
        try:
            # CLI validation primarily checks template structure and basic issues
            problems = generator.detect_and_fix_problems(generator.output_data.copy(), generator.template)
            if problems:
                console.print("[bold yellow]Validation Issues Found:[/bold yellow]")
                for p in problems:
                    console.print(f"- {p}")
            else:
                console.print("[bold green]Dataset passed basic validation checks.[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error during validation: {str(e)}[/bold red]")

    if args.split:
        if not args.split_output_paths or len(args.split_output_paths) != 3:
            console.print("[bold red]Error: --split requires --split-output-paths TRAIN_PATH VALID_PATH TEST_PATH.[/bold red]")
            sys.exit(1)
        try:
            train_data, valid_data, test_data = generator.split_dataset(
                generator.output_data,
                train_ratio=args.split[0],
                valid_ratio=args.split[1],
                test_ratio=args.split[2]
            )
            split_output_format = args.output_format or "jsonl" # Default for splits
            generator.save_data(train_data, args.split_output_paths[0], split_output_format)
            if valid_data:
                generator.save_data(valid_data, args.split_output_paths[1], split_output_format)
            if test_data:
                generator.save_data(test_data, args.split_output_paths[2], split_output_format)
            console.print(f"[bold green]Dataset split and saved to {args.split_output_paths[0]}, {args.split_output_paths[1]}, {args.split_output_paths[2]}[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error during splitting: {str(e)}[/bold red]")
            sys.exit(1)

    if args.visualize:
        try:
            generator.visualize_data(generator.output_data, output_path=args.visualize)
            console.print(f"[bold green]Visualizations saved to {args.visualize}[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error generating visualizations: {str(e)}[/bold red]")

    # --- Data Saving ---
    final_output_data = generator.output_data

    if args.output:
        output_format = args.output_format or os.path.splitext(args.output)[1].lower()[1:]
        if output_format.startswith('.'): output_format = output_format[1:]
        try:
            generator.save_data(final_output_data, args.output, output_format)
        except Exception as e:
            console.print(f"[bold red]Error saving output data: {str(e)}[/bold red]")
            sys.exit(1)
    elif args.create_records and args.create_output:
        output_format = args.output_format or os.path.splitext(args.create_output)[1].lower()[1:]
        if output_format.startswith('.'): output_format = output_format[1:]
        try:
            generator.save_data(final_output_data, args.create_output, output_format)
        except Exception as e:
            console.print(f"[bold red]Error saving created data: {str(e)}[/bold red]")
            sys.exit(1)
    elif not args.split and not args.visualize: # If no explicit output or visualization, display sample
        if final_output_data:
            console.print("\n[bold]Processed Data Sample:[/bold]")
            display_dataset_sample(final_output_data)
        else:
            console.print("[bold yellow]No data processed or generated.[/bold yellow]")


if __name__ == "__main__":
    try:
        # Check for command line arguments. If none, run GUI.
        # If arguments are present, run CLI mode.
        if len(sys.argv) > 1 and '--interactive' not in sys.argv:
            cli_mode()
        else:
            app = QApplication(sys.argv)
            main_window = MainWindow()
            main_window.show()
            sys.exit(app.exec_())
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Operation canceled by user.[/bold yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {str(e)}[/bold red]")
        logger.exception("Unhandled exception in main execution:")
        sys.exit(1)

