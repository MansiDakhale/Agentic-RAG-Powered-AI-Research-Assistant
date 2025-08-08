# utils/document_loader.py
import streamlit as st
from typing import List, Dict
import PyPDF2
import docx
import io
from datetime import datetime

class DocumentLoader:
    """Handles loading and processing of various document types"""
    
    def __init__(self):
        self.supported_types = {'.pdf', '.txt', '.docx'}
        self.chunk_size = 1000
        self.chunk_overlap = 200
    
    def process_files(self, uploaded_files) -> List[str]:
        """Process uploaded files and return document chunks"""
        all_documents = []
        
        for uploaded_file in uploaded_files:
            try:
                # Get file extension
                file_extension = uploaded_file.name.lower().split('.')[-1]
                
                if f'.{file_extension}' not in self.supported_types:
                    st.warning(f"Unsupported file type: {uploaded_file.name}")
                    continue
                
                # Process based on file type
                text_content = self._extract_text(uploaded_file, file_extension)
                
                if text_content:
                    # Chunk the document
                    chunks = self._chunk_text(text_content, uploaded_file.name)
                    all_documents.extend(chunks)
                    
                    st.info(f"✅ Processed {uploaded_file.name}: {len(chunks)} chunks")
                else:
                    st.warning(f"No text content found in {uploaded_file.name}")
                    
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        return all_documents
    
    def _extract_text(self, uploaded_file, file_extension: str) -> str:
        """Extract text from uploaded file"""
        
        if file_extension == 'pdf':
            return self._extract_from_pdf(uploaded_file)
        elif file_extension == 'txt':
            return self._extract_from_txt(uploaded_file)
        elif file_extension == 'docx':
            return self._extract_from_docx(uploaded_file)
        else:
            return ""
    
    def _extract_from_pdf(self, uploaded_file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            text_content = ""
            
            for page in pdf_reader.pages:
                text_content += page.extract_text() + "\n"
            
            return text_content
        except Exception as e:
            st.error(f"PDF extraction error: {str(e)}")
            return ""
    
    def _extract_from_txt(self, uploaded_file) -> str:
        """Extract text from TXT file"""
        try:
            return str(uploaded_file.read(), "utf-8")
        except Exception as e:
            st.error(f"TXT extraction error: {str(e)}")
            return ""
    
    def _extract_from_docx(self, uploaded_file) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(io.BytesIO(uploaded_file.read()))
            text_content = ""
            
            for paragraph in doc.paragraphs:
                text_content += paragraph.text + "\n"
            
            return text_content
        except Exception as e:
            st.error(f"DOCX extraction error: {str(e)}")
            return ""
    
    def _chunk_text(self, text: str, filename: str) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        
        # Simple sentence-based chunking
        sentences = text.replace('\n', ' ').split('. ')
        
        current_chunk = ""
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_size = len(sentence)
            
            # If adding this sentence would exceed chunk size, start new chunk
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunks.append(f"[Source: {filename}] " + current_chunk.strip())
                
                # Start new chunk with overlap
                overlap_text = current_chunk.split()[-self.chunk_overlap//10:]  # Rough overlap
                current_chunk = " ".join(overlap_text) + " " + sentence + ". "
                current_size = len(current_chunk)
            else:
                current_chunk += sentence + ". "
                current_size += sentence_size + 2
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(f"[Source: {filename}] " + current_chunk.strip())
        
        return chunks

# Test the document loader
if __name__ == "__main__":
    loader = DocumentLoader()
    
    # Create test text file
    test_text = """
    This is a sample document for testing the document loader functionality.
    It contains multiple sentences that should be properly chunked.
    The chunking algorithm should preserve context while maintaining reasonable chunk sizes.
    This helps ensure that the vector search can find relevant information effectively.
    """
    
    # Save test file
    with open("test_document.txt", "w") as f:
        f.write(test_text)
    
    print("Document loader test completed!")