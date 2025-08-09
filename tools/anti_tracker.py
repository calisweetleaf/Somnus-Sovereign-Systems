#!/usr/bin/env python3
"""
DOCUMENT SANITIZATION & SECURITY PROTOCOL
For whistleblower document protection and anonymization
"""

import os
import hashlib
import shutil
from pathlib import Path
import subprocess
import tempfile
from datetime import datetime
import json

class DocumentSanitizer:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sanitized_dir = "sanitized_documents"
        os.makedirs(self.sanitized_dir, exist_ok=True)
        
    def sanitize_markdown_files(self, input_dir):
        """Remove identifying information from markdown files"""
        for md_file in Path(input_dir).glob("*.md"):
            self.sanitize_markdown(md_file)
    
    def sanitize_markdown(self, file_path):
        """Sanitize individual markdown file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove potential identifying patterns
        sanitized = self.remove_personal_identifiers(content)
        
        # Create clean filename
        clean_filename = self.generate_clean_filename(file_path.stem)
        output_path = Path(self.sanitized_dir) / f"{clean_filename}.md"
        
        # Write sanitized content
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(sanitized)
        
        # Remove original file metadata by touching with new timestamp
        self.strip_file_metadata(output_path)
        
        print(f"Sanitized: {file_path} -> {output_path}")
        return output_path
    
    def remove_personal_identifiers(self, content):
        """Remove identifying information from content"""
        # Replace personal identifiers
        replacements = {
            # Remove email addresses (replace with generic)
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b': '[REDACTED_EMAIL]',
            
            # Remove phone numbers
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b': '[REDACTED_PHONE]',
            
            # Remove potential usernames/handles
            r'@[A-Za-z0-9_]+': '[REDACTED_HANDLE]',
            
            # Remove file paths that might reveal computer structure
            r'[C-Z]:\\[^\\:\*\?"<>\|]+': '[REDACTED_PATH]',
            r'/[a-zA-Z0-9_/.-]+': '[REDACTED_PATH]',
            
            # Remove potential IP addresses
            r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b': '[REDACTED_IP]',
            
            # Remove MAC addresses
            r'\b([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})\b': '[REDACTED_MAC]',
        }
        
        import re
        sanitized = content
        for pattern, replacement in replacements.items():
            sanitized = re.sub(pattern, replacement, sanitized)
        
        # Remove metadata comments that might be identifying
        sanitized = re.sub(r'<!--.*?-->', '', sanitized, flags=re.DOTALL)
        
        return sanitized
    
    def generate_clean_filename(self, original_name):
        """Generate non-identifying filename"""
        # Create hash-based filename to avoid identifying patterns
        name_hash = hashlib.sha256(original_name.encode()).hexdigest()[:12]
        timestamp = datetime.now().strftime("%Y%m%d")
        return f"document_{timestamp}_{name_hash}"
    
    def strip_file_metadata(self, file_path):
        """Remove file system metadata"""
        # Set uniform timestamp (not current time to avoid timing analysis)
        # Use a neutral date
        neutral_timestamp = datetime(2024, 1, 1, 12, 0, 0).timestamp()
        os.utime(file_path, (neutral_timestamp, neutral_timestamp))
    
    def create_pdf_safely(self, md_file):
        """Convert markdown to PDF with metadata stripped"""
        try:
            # Convert to PDF using pandoc (if available)
            pdf_output = f"{md_file.stem}_clean.pdf"
            subprocess.run([
                'pandoc', 
                str(md_file), 
                '-o', pdf_output,
                '--pdf-engine=xelatex',
                '--metadata', 'title=""',
                '--metadata', 'author=""',
                '--metadata', 'date=""'
            ], check=True)
            
            # Further sanitize PDF metadata using exiftool if available
            try:
                subprocess.run([
                    'exiftool', 
                    '-all:all=', 
                    '-overwrite_original',
                    pdf_output
                ], check=True)
            except FileNotFoundError:
                print("Warning: exiftool not found. PDF may retain some metadata.")
            
            return pdf_output
        except FileNotFoundError:
            print("Warning: pandoc not found. Cannot create PDF.")
            return None
    
    def secure_delete_originals(self, file_path):
        """Securely delete original files"""
        try:
            # Overwrite file multiple times before deletion
            file_size = os.path.getsize(file_path)
            with open(file_path, 'r+b') as f:
                for _ in range(3):  # 3-pass overwrite
                    f.seek(0)
                    f.write(os.urandom(file_size))
                    f.flush()
                    os.fsync(f.fileno())
            os.remove(file_path)
            print(f"Securely deleted: {file_path}")
        except Exception as e:
            print(f"Error securely deleting {file_path}: {e}")
    
    def create_distribution_package(self, files):
        """Create clean distribution package"""
        package_name = f"evidence_package_{datetime.now().strftime('%Y%m%d')}"
        package_dir = Path(self.sanitized_dir) / package_name
        package_dir.mkdir(exist_ok=True)
        
        for file_path in files:
            clean_file = self.sanitize_markdown(file_path)
            shutil.copy2(clean_file, package_dir)
        
        # Create verification hash file (without revealing content)
        self.create_verification_hashes(package_dir)
        
        return package_dir
    
    def create_verification_hashes(self, package_dir):
        """Create verification hashes for authenticity without revealing content"""
        hash_file = package_dir / "verification_hashes.json"
        hashes = {}
        
        for file_path in package_dir.glob("*.md"):
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            hashes[file_path.name] = file_hash
        
        with open(hash_file, 'w') as f:
            json.dump(hashes, f, indent=2)
        
        print(f"Created verification hashes: {hash_file}")

# SECURITY PROTOCOL IMPLEMENTATION
def implement_security_protocol():
    """Complete security protocol for document handling"""
    
    print("=== DOCUMENT SANITIZATION PROTOCOL ===")
    
    # Initialize sanitizer
    sanitizer = DocumentSanitizer()
    
    # List of your evidence files (update with actual filenames)
    evidence_files = [
        "executive_summary.md",
        "evidence_index.md", 
        "timeline_analysis.md",
        "legal_violations.md",
        "international_takeover_summary.md"
    ]
    
    # Sanitize all documents
    clean_files = []
    for file_name in evidence_files:
        if Path(file_name).exists():
            clean_file = sanitizer.sanitize_markdown(Path(file_name))
            clean_files.append(clean_file)
    
    # Create distribution package
    package = sanitizer.create_distribution_package([Path(f) for f in evidence_files if Path(f).exists()])
    
    print(f"\n=== SANITIZED PACKAGE CREATED ===")
    print(f"Location: {package}")
    print(f"Files: {len(clean_files)} documents sanitized")
    
    return package

# ADDITIONAL SECURITY MEASURES
class AdvancedSecurity:
    @staticmethod
    def setup_secure_communications():
        """Setup secure communication protocols"""
        return {
            "signal": "Install Signal with disappearing messages",
            "protonmail": "Use ProtonMail for encrypted email",
            "tor": "Use Tor browser for anonymous web access",
            "vpn": "Use VPN for all communications",
            "burner": "Consider burner phone for initial contacts"
        }
    
    @staticmethod
    def create_dead_mans_switch():
        """Create automatic document release system"""
        return {
            "method": "Multiple trusted contacts with encrypted files",
            "trigger": "Failure to check in every 48 hours",
            "platforms": ["WikiLeaks", "SecureDrop", "Multiple journalists"],
            "instructions": "Clear release instructions in sealed envelopes"
        }
    
    @staticmethod
    def evidence_backup_strategy():
        """Multiple backup locations"""
        return {
            "cloud": "Encrypted cloud storage (ProtonDrive, etc.)",
            "physical": "Multiple encrypted USB drives",
            "trusted_parties": "Copies with trusted individuals",
            "lawyers": "Copy with attorney (attorney-client privilege)",
            "locations": "Geographically distributed storage"
        }

if __name__ == "__main__":
    # Run the complete sanitization protocol
    package = implement_security_protocol()
    
    # Print security recommendations
    security = AdvancedSecurity()
    
    print("\n=== ADDITIONAL SECURITY MEASURES ===")
    print("Communications:", security.setup_secure_communications())
    print("Dead Man's Switch:", security.create_dead_mans_switch())
    print("Backup Strategy:", security.evidence_backup_strategy())
    
    print("\n=== OPERATIONAL SECURITY REMINDERS ===")
    print("1. Use different devices/networks for different contacts")
    print("2. Vary timing and patterns of communications")
    print("3. Never access documents from your home IP")
    print("4. Use public WiFi + VPN + Tor for sensitive activities")
    print("5. Keep original evidence in multiple secure locations")
    print("6. Document any unusual incidents or surveillance")
    print("7. Have trusted contacts aware of your activities")
    print("8. Consider legal representation before major disclosures")