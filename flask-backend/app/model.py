import logging
import spacy
from spacy.matcher import Matcher
import re
from datetime import datetime
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
import cv2
import numpy as np
import pandas as pd
import os
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document processing and text extraction"""

    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.matcher = Matcher(self.nlp.vocab)
            self._setup_matcher_patterns()
        except Exception as e:
            logger.error(f"Error initializing NLP model: {e}")
            raise

    def _setup_matcher_patterns(self):
        """Setup spaCy matcher patterns"""
        patterns = {
            "INTEREST_RATE": [
                {"LOWER": "interest"}, {"LOWER": "rate"},
                {"IS_PUNCT": True}, {"LIKE_NUM": True}, {"LOWER": "%"}
            ],
            "PRINCIPAL_AMOUNT": [
                {"LOWER": "principal"}, {"LOWER": "amount"},
                {"IS_PUNCT": True}, {"ENT_TYPE": "MONEY"}
            ],
            "COVENANT": [
                {"LEMMA": "shall"}, {"LOWER": "maintain"}
            ],
            "DEFAULT": [
                {"LOWER": "default"}, {"LOWER": "occurs"}
            ]
        }
        for name, pattern in patterns.items():
            self.matcher.add(name, [pattern])

    def process_text(self, file_content: str) -> str:
        """Simulate text extraction from file content."""
        return file_content

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        try:
            text = ""
            # Use PdfReader for text-based extraction
            with open(pdf_path, "rb") as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text()

            if text.strip():
                return text

            # Fallback to OCR for image-based PDFs
            images = convert_from_path(pdf_path, dpi=300)
            for img in images:
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
                _, thresh = cv2.threshold(img_cv, 150, 255, cv2.THRESH_BINARY)
                text += pytesseract.image_to_string(thresh)
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            raise

    def extract_text_from_image(self, img_path: str) -> str:
        """Handle JPG/PNG images with OCR."""
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not read image: {img_path}")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            return pytesseract.image_to_string(thresh)
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {e}")
            raise

    def extract_text_from_excel(self, excel_path: str) -> str:
        """Handle both XLS and XLSX formats."""
        try:
            df = pd.read_excel(excel_path, sheet_name=None)
            text = ""
            for sheet_name, sheet_df in df.items():
                text += f"\n\n[{sheet_name}]\n{sheet_df.to_string(index=False)}"
            return text
        except Exception as e:
            logger.error(f"Error processing Excel file {excel_path}: {e}")
            raise

    def extract_text_from_csv(self, csv_path: str) -> str:
        """Handle CSV files."""
        try:
            df = pd.read_csv(csv_path)
            return df.to_string(index=False)
        except Exception as e:
            logger.error(f"Error processing CSV file {csv_path}: {e}")
            raise

    def process_any_file(self, file_path: str) -> str:
        """Automatically detect file type and extract text."""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            if file_path.endswith('.pdf'):
                return self.extract_text_from_pdf(file_path)
            elif file_path.endswith(('.jpg', '.jpeg', '.png')):
                return self.extract_text_from_image(file_path)
            elif file_path.endswith(('.csv', '.txt')):
                return self.extract_text_from_csv(file_path)
            elif file_path.endswith(('.xls', '.xlsx')):
                return self.extract_text_from_excel(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_path}")
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise

class TermExtractor:
    """Extracts and categorizes terms from text"""

    def __init__(self, nlp, matcher):
        self.nlp = nlp
        self.matcher = matcher

    def extract_clauses(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Extract and categorize clauses from text"""
        try:
            doc = self.nlp(text)
            clauses = {
                "financial_terms": {},
                "legal_compliance": {},
                "operational": {},
                "regulatory": {},
                "boilerplate": {}
            }

            # Extract financial terms
            self._extract_financial_terms(doc, clauses)

            # Extract legal compliance terms
            self._extract_legal_compliance(doc, clauses)

            # Extract operational terms
            self._extract_operational_terms(doc, clauses)

            # Extract regulatory terms
            self._extract_regulatory_terms(doc, clauses)

            # Extract boilerplate terms
            self._extract_boilerplate_terms(doc, clauses)

            return clauses
        except Exception as e:
            logger.error(f"Error extracting clauses: {e}")
            raise

    def _extract_financial_terms(self, doc, clauses):
        """Extract financial terms"""
        interest_rate_pattern = re.search(r"interest rate:?\s*(\d+(?:\.\d+)?)\s*%", doc.text.lower())
        if interest_rate_pattern:
            clauses["financial_terms"]["interest_rate"] = interest_rate_pattern.group(1)

    def _extract_legal_compliance(self, doc, clauses):
        """Extract legal compliance terms"""
        # Representations and Warranties
        rep_warranty_keywords = ["represents", "warrants", "warranties"]
        for sent in doc.sents:
            if any(keyword in sent.text.lower() for keyword in rep_warranty_keywords):
                clauses["legal_compliance"]["representations_warranties"] = sent.text

    def _extract_operational_terms(self, doc, clauses):
        """Extract operational terms"""
        # Prepayment Terms
        prepayment_match = re.search(r"prepayment (.*?\.)", doc.text.lower())
        if prepayment_match:
            clauses["operational"]["prepayment_terms"] = prepayment_match.group(1)

    def _extract_regulatory_terms(self, doc, clauses):
        """Extract regulatory terms"""
        # KYC/AML Compliance
        kyc_match = re.search(r"kyc/aml compliance: (.*?\.)", doc.text.lower())
        if kyc_match:
            clauses["regulatory"]["kyc_aml"] = kyc_match.group(1)

    def _extract_boilerplate_terms(self, doc, clauses):
        """Extract boilerplate terms"""
        # Confidentiality
        confidentiality_sent = [sent.text for sent in doc.sents if "confidential" in sent.text.lower()]
        if confidentiality_sent:
            clauses["boilerplate"]["confidentiality"] = confidentiality_sent[0]

class TermValidator:
    """Validates extracted terms against business rules"""

    def validate_terms(self, terms):
        """Validate extracted terms"""
        validation_results = {}
        try:
            self._validate_financial_terms(terms, validation_results)
            return validation_results
        except Exception as e:
            logger.error(f"Error validating terms: {e}")
            raise

    def _validate_financial_terms(self, terms, validation_results):
        """Validate financial terms"""
        if "financial_terms" in terms:
            financial_terms = terms["financial_terms"]

            # Validate interest rate
            if "interest_rate" in financial_terms:
                try:
                    rate = float(financial_terms["interest_rate"])
                    is_valid = 0 <= rate <= 20
                    validation_results["interest_rate"] = {
                        "valid": is_valid,
                        "issue": "" if is_valid else "Interest rate outside expected range (0-20%)"
                    }
                except ValueError:
                    validation_results["interest_rate"] = {
                        "valid": False,
                        "issue": "Could not parse interest rate"
                    }

            # Validate principal amount
            if "principal_amount" in financial_terms:
                try:
                    amount = float(financial_terms["principal_amount"].replace(",", ""))
                    is_valid = amount > 0 and amount < 1_000_000_000_000
                    validation_results["principal_amount"] = {
                        "valid": is_valid,
                        "confidence": 0.9 if is_valid else 0.6,
                        "issue": "" if is_valid else "Principal amount seems unreasonable"
                    }
                except ValueError:
                    validation_results["principal_amount"] = {
                        "valid": False,
                        "confidence": 0.4,
                        "issue": "Could not parse principal amount"
                    }

class Model:
    """Main model class to process documents and validate terms"""

    def __init__(self):
        self.processor = DocumentProcessor()
        self.extractor = TermExtractor(self.processor.nlp, self.processor.matcher)
        self.validator = TermValidator()

    def process_document(self, file_path: str):
        """Process the document content from a file."""
        try:
            # Step 1: Extract text from the file
            text = self.processor.process_any_file(file_path)
            logger.info("Text extraction completed.")

            # Step 2: Extract terms
            terms = self.extractor.extract_clauses(text)
            logger.info("Term extraction completed.")

            # Step 3: Validate terms
            validation_results = self.validator.validate_terms(terms)
            logger.info("Validation completed.")

            return {"terms": terms, "validation_results": validation_results}
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise