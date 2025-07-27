#!/usr/bin/env python3
"""Advanced Web Interface for PDF Outline Extractor"""
import os, json, tempfile, sys
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string

sys.path.append('/app/src')
sys.path.append('src')

try:
    from main import extract_pdf_outline
    print("✅ Successfully imported advanced extract_pdf_outline")
except ImportError as e:
    print(f"⚠️ Failed to import main.py: {e}")
    def extract_pdf_outline(pdf_path, output_path): 
        return False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced PDF Outline Extractor</title>
    <style>
        * {margin:0;padding:0;box-sizing:border-box;}
        body {font-family:'Segoe UI',sans-serif;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);min-height:100vh;padding:20px;}
        .container {max-width:1400px;margin:0 auto;}
        .header {text-align:center;margin-bottom:40px;color:white;}
        .header h1 {font-size:2.8rem;margin-bottom:10px;text-shadow:2px 2px 4px rgba(0,0,0,0.3);}
        .header p {font-size:1.2rem;opacity:0.9;margin-bottom:10px;}
        .features-badge {background:rgba(255,255,255,0.2);padding:8px 16px;border-radius:20px;font-size:0.9rem;margin:5px;}
        .main-content {display:grid;grid-template-columns:1fr 1fr;gap:30px;margin-bottom:30px;}
        @media (max-width:768px) {.main-content {grid-template-columns:1fr;}}
        .card {background:white;border-radius:15px;padding:30px;box-shadow:0 10px 30px rgba(0,0,0,0.2);transition:transform 0.3s ease;}
        .card:hover {transform:translateY(-5px);}
        .upload-area {border:3px dashed #667eea;border-radius:10px;padding:40px 20px;text-align:center;cursor:pointer;transition:all 0.3s ease;background:#f8f9ff;}
        .upload-area:hover {border-color:#764ba2;background:#f0f4ff;}
        .upload-area.dragover {border-color:#5a67d8;background:#e6fffa;transform:scale(1.02);}
        .upload-icon {font-size:3rem;color:#667eea;margin-bottom:15px;}
        .upload-text {font-size:1.1rem;color:#555;margin-bottom:10px;}
        .upload-hint {font-size:0.9rem;color:#888;}
        #fileInput {display:none;}
        .btn {background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;border:none;padding:12px 30px;border-radius:25px;cursor:pointer;font-size:1rem;font-weight:600;transition:all 0.3s ease;margin-top:15px;}
        .btn:hover {transform:translateY(-2px);box-shadow:0 5px 15px rgba(102,126,234,0.4);}
        .btn:disabled {opacity:0.6;cursor:not-allowed;transform:none;}
        .processing {display:none;text-align:center;margin:20px 0;}
        .spinner {width:40px;height:40px;border:4px solid #f3f3f3;border-top:4px solid #667eea;border-radius:50%;animation:spin 1s linear infinite;margin:0 auto 15px;}
        @keyframes spin {0% {transform:rotate(0deg);} 100% {transform:rotate(360deg);}}
        .results {display:none;}
        .results-header {display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;padding-bottom:15px;border-bottom:2px solid #f0f0f0;}
        .results-title {color:#333;font-size:1.3rem;font-weight:600;}
        .stats {display:flex;gap:15px;flex-wrap:wrap;}
        .stat {background:#f8f9ff;padding:8px 15px;border-radius:20px;font-size:0.9rem;color:#667eea;font-weight:600;}
        .metadata {background:#e6fffa;border:1px solid #81e6d9;border-radius:8px;padding:15px;margin:15px 0;}
        .metadata h4 {color:#2d3748;margin-bottom:10px;}
        .metadata-item {display:flex;justify-content:space-between;margin:5px 0;font-size:0.9rem;}
        .language-indicator {background:#38a169;color:white;padding:4px 8px;border-radius:12px;font-size:0.8rem;text-transform:uppercase;}
        .document-title {background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:15px 20px;border-radius:10px;margin-bottom:20px;font-size:1.1rem;font-weight:600;}
        .outline-item {margin:10px 0;padding:12px 15px;background:#f9f9f9;border-radius:8px;border-left:4px solid #667eea;transition:all 0.3s ease;}
        .outline-item:hover {background:#f0f4ff;transform:translateX(5px);}
        .outline-item.h1 {border-left-color:#e53e3e;font-weight:600;font-size:1.1rem;background:#fff5f5;}
        .outline-item.h2 {border-left-color:#3182ce;margin-left:20px;font-weight:500;background:#f7fafc;}
        .outline-item.h3 {border-left-color:#38a169;margin-left:40px;font-size:0.95rem;background:#f0fff4;}
        .page-number {float:right;background:#667eea;color:white;padding:2px 8px;border-radius:10px;font-size:0.8rem;font-weight:600;}
        .empty-state {text-align:center;color:#888;padding:40px 20px;}
        .empty-icon {font-size:4rem;margin-bottom:15px;opacity:0.5;}
        .download-btn {background:#38a169;margin-top:20px;}
        .download-btn:hover {box-shadow:0 5px 15px rgba(56,161,105,0.4);}
        .file-info {display:none;background:#e6fffa;border:1px solid #81e6d9;border-radius:8px;padding:15px;margin-top:15px;}
        .file-name {font-weight:600;color:#2d3748;margin-bottom:5px;}
        .file-size {color:#4a5568;font-size:0.9rem;}
        .error-message {display:none;background:#fed7d7;border:1px solid #fc8181;color:#c53030;padding:15px;border-radius:8px;margin-top:15px;}
        .alert {padding:12px 16px;border-radius:8px;margin:10px 0;font-weight:500;}
        .alert-success {background:#c6f6d5;border:1px solid #9ae6b4;color:#22543d;}
        .features-showcase {background:white;border-radius:15px;padding:30px;box-shadow:0 10px 30px rgba(0,0,0,0.2);margin-top:30px;}
        .features-showcase h3 {color:#333;margin-bottom:20px;text-align:center;font-size:1.5rem;}
        .features-grid {display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px;}
        .feature-card {padding:20px;background:#f8f9ff;border-radius:12px;border:1px solid #e2e8f0;}
        .feature-icon {background:#667eea;color:white;width:40px;height:40px;border-radius:50%;display:flex;align-items:center;justify-content:center;margin-bottom:15px;font-size:1.2rem;}
        .feature-title {font-weight:600;color:#2d3748;margin-bottom:10px;}
        .feature-desc {color:#4a5568;font-size:0.9rem;line-height:1.4;}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Advanced PDF Outline Extractor</h1>
            <p>AI-powered heading detection with language awareness & noise filtering</p>
            <div>
                <span class="features-badge">🌍 Multi-Language</span>
                <span class="features-badge">🧠 Smart Detection</span>
                <span class="features-badge">🔧 Noise Filtering</span>
                <span class="features-badge">⚡ Fast Processing</span>
            </div>
        </div>
        
        <div class="main-content">
            <div class="card">
                <h2 style="margin-bottom:20px;color:#333;">📤 Upload PDF Document</h2>
                <div class="upload-area" id="uploadArea">
                    <div class="upload-icon">📁</div>
                    <div class="upload-text">Click or drag PDF file here</div>
                    <div class="upload-hint">Supports PDF files up to 50MB • Advanced processing for complex documents</div>
                </div>
                <input type="file" id="fileInput" accept=".pdf"/>
                <div class="file-info" id="fileInfo">
                    <div class="file-name" id="fileName"></div>
                    <div class="file-size" id="fileSize"></div>
                </div>
                <button class="btn" id="processBtn" disabled>🚀 Extract Advanced Outline</button>
                
                <div class="processing" id="processing">
                    <div class="spinner"></div>
                    <div>Processing with advanced AI algorithms...</div>
                </div>
                
                <div class="error-message" id="errorMessage"></div>
            </div>
            
            <div class="card">
                <div class="results" id="results">
                    <div class="results-header">
                        <div class="results-title">📋 Extracted Document Outline</div>
                        <div class="stats">
                            <div class="stat" id="headingCount">0 headings</div>
                            <div class="stat" id="pageCount">0 pages</div>
                            <div class="stat" id="processingTime">0s</div>
                        </div>
                    </div>
                    
                    <div class="document-title" id="documentTitle"></div>
                    
                    <div class="metadata" id="metadata">
                        <h4>📊 Analysis Metadata</h4>
                        <div class="metadata-item">
                            <span>Detected Language:</span>
                            <span class="language-indicator" id="detectedLanguage">Unknown</span>
                        </div>
                        <div class="metadata-item">
                            <span>Text Blocks Analyzed:</span>
                            <span id="blocksAnalyzed">0</span>
                        </div>
                        <div class="metadata-item">
                            <span>Processing Time:</span>
                            <span id="processingTimeDetail">0s</span>
                        </div>
                    </div>
                    
                    <div class="outline-container" id="outlineContainer"></div>
                    <button class="btn download-btn" id="downloadBtn">💾 Download JSON with Metadata</button>
                </div>
                
                <div class="empty-state" id="emptyState">
                    <div class="empty-icon">📄</div>
                    <div>Upload a PDF to see the advanced AI-powered outline extraction</div>
                </div>
            </div>
        </div>
        
        <div class="features-showcase">
            <h3>🚀 Advanced Features</h3>
            <div class="features-grid">
                <div class="feature-card">
                    <div class="feature-icon">🌍</div>
                    <div class="feature-title">Multi-Language Detection</div>
                    <div class="feature-desc">Automatically detects document language including English, Spanish, French, German, Japanese, Chinese, and Korean with specialized heading patterns.</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🧠</div>
                    <div class="feature-title">AI-Powered Heading Detection</div>
                    <div class="feature-desc">Advanced scoring algorithm analyzes font size, weight, position, and semantic patterns to accurately identify H1, H2, and H3 headings.</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🧹</div>
                    <div class="feature-title">Smart Noise Filtering</div>
                    <div class="feature-desc">Filters out headers, footers, page numbers, form fields, and table of contents entries using pattern recognition.</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🔧</div>
                    <div class="feature-title">Text Cleaning & OCR Repair</div>
                    <div class="feature-desc">Repairs common OCR errors, fixes fragmented words, removes corruption patterns, and normalizes text formatting.</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📊</div>
                    <div class="feature-title">Statistical Analysis</div>
                    <div class="feature-desc">Calculates font statistics, heading thresholds, and provides detailed metadata about the document analysis process.</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">⚡</div>
                    <div class="feature-title">High Performance</div>
                    <div class="feature-desc">Processes up to 50 pages in under 10 seconds with optimized algorithms and efficient text extraction.</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        class AdvancedPDFExtractor {
            constructor() {
                this.initializeElements();
                this.attachEventListeners();
                this.currentFile = null;
                this.currentResults = null;
            }
            
            initializeElements() {
                this.uploadArea = document.getElementById('uploadArea');
                this.fileInput = document.getElementById('fileInput');
                this.fileInfo = document.getElementById('fileInfo');
                this.fileName = document.getElementById('fileName');
                this.fileSize = document.getElementById('fileSize');
                this.processBtn = document.getElementById('processBtn');
                this.processing = document.getElementById('processing');
                this.results = document.getElementById('results');
                this.emptyState = document.getElementById('emptyState');
                this.documentTitle = document.getElementById('documentTitle');
                this.outlineContainer = document.getElementById('outlineContainer');
                this.headingCount = document.getElementById('headingCount');
                this.pageCount = document.getElementById('pageCount');
                this.processingTime = document.getElementById('processingTime');
                this.downloadBtn = document.getElementById('downloadBtn');
                this.errorMessage = document.getElementById('errorMessage');
                this.metadata = document.getElementById('metadata');
                this.detectedLanguage = document.getElementById('detectedLanguage');
                this.blocksAnalyzed = document.getElementById('blocksAnalyzed');
                this.processingTimeDetail = document.getElementById('processingTimeDetail');
            }
            
            attachEventListeners() {
                this.uploadArea.addEventListener('click', () => this.fileInput.click());
                this.uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
                this.uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
                this.uploadArea.addEventListener('drop', this.handleDrop.bind(this));
                this.fileInput.addEventListener('change', this.handleFileSelect.bind(this));
                this.processBtn.addEventListener('click', this.processFile.bind(this));
                this.downloadBtn.addEventListener('click', this.downloadResults.bind(this));
            }
            
            handleDragOver(e) {
                e.preventDefault();
                this.uploadArea.classList.add('dragover');
            }
            
            handleDragLeave(e) {
                e.preventDefault();
                this.uploadArea.classList.remove('dragover');
            }
            
            handleDrop(e) {
                e.preventDefault();
                this.uploadArea.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    this.handleFile(files[0]);
                }
            }
            
            handleFileSelect(e) {
                const file = e.target.files[0];
                if (file) {
                    this.handleFile(file);
                }
            }
            
            handleFile(file) {
                if (!file.type.includes('pdf')) {
                    this.showError('Please select a PDF file');
                    return;
                }
                if (file.size > 50 * 1024 * 1024) {
                    this.showError('File size must be less than 50MB');
                    return;
                }
                this.currentFile = file;
                this.showFileInfo(file);
                this.processBtn.disabled = false;
                this.hideError();
            }
            
            showFileInfo(file) {
                this.fileName.textContent = file.name;
                this.fileSize.textContent = this.formatFileSize(file.size);
                this.fileInfo.style.display = 'block';
            }
            
            formatFileSize(bytes) {
                if (bytes === 0) return '0 Bytes';
                const k = 1024;
                const sizes = ['Bytes', 'KB', 'MB', 'GB'];
                const i = Math.floor(Math.log(bytes) / Math.log(k));
                return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
            }
            
            async processFile() {
                if (!this.currentFile) return;
                this.showProcessing();
                
                try {
                    const formData = new FormData();
                    formData.append('pdf', this.currentFile);
                    
                    const response = await fetch('/api/extract', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.error || 'Failed to process PDF');
                    }
                    
                    const results = await response.json();
                    this.showResults(results);
                    
                } catch (error) {
                    this.showError('Failed to process PDF: ' + error.message);
                } finally {
                    this.hideProcessing();
                }
            }
            
            showProcessing() {
                this.processBtn.disabled = true;
                this.processing.style.display = 'block';
                this.results.style.display = 'none';
                this.emptyState.style.display = 'none';
            }
            
            hideProcessing() {
                this.processBtn.disabled = false;
                this.processing.style.display = 'none';
            }
            
            showResults(data) {
                this.currentResults = data;
                this.documentTitle.textContent = `📄 ${data.title}`;
                this.headingCount.textContent = `${data.outline.length} headings`;
                
                const maxPage = data.outline.length > 0 ? Math.max(...data.outline.map(h => h.page)) : 1;
                this.pageCount.textContent = `${maxPage} pages`;
                
                // Show metadata if available
                if (data.metadata) {
                    this.detectedLanguage.textContent = data.metadata.detected_language || 'Unknown';
                    this.blocksAnalyzed.textContent = data.metadata.total_blocks_analyzed || 'N/A';
                    this.processingTimeDetail.textContent = data.metadata.processing_time_seconds ? 
                        `${data.metadata.processing_time_seconds}s` : 'N/A';
                    this.processingTime.textContent = data.metadata.processing_time_seconds ? 
                        `${data.metadata.processing_time_seconds}s` : '0s';
                    this.metadata.style.display = 'block';
                } else {
                    this.metadata.style.display = 'none';
                }
                
                this.renderOutline(data.outline);
                this.results.style.display = 'block';
                this.emptyState.style.display = 'none';
                this.showSuccess('Advanced outline extraction completed successfully!');
            }
            
            renderOutline(outline) {
                this.outlineContainer.innerHTML = '';
                
                if (outline.length === 0) {
                    this.outlineContainer.innerHTML = '<div style="text-align:center;color:#666;padding:20px;">No headings detected by the AI algorithm.</div>';
                    return;
                }
                
                outline.forEach(item => {
                    const div = document.createElement('div');
                    div.className = `outline-item ${item.level.toLowerCase()}`;
                    div.innerHTML = `${item.text} <span class="page-number">p. ${item.page}</span>`;
                    this.outlineContainer.appendChild(div);
                });
            }
            
            downloadResults() {
                if (!this.currentResults) return;
                
                const blob = new Blob([JSON.stringify(this.currentResults, null, 2)], {
                    type: 'application/json'
                });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `${this.currentResults.title.replace(/[^a-z0-9]/gi, '_')}_advanced_outline.json`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                this.showSuccess('Advanced outline with metadata downloaded!');
            }
            
            showError(message) {
                this.errorMessage.textContent = message;
                this.errorMessage.style.display = 'block';
                setTimeout(() => this.hideError(), 5000);
            }
            
            hideError() {
                this.errorMessage.style.display = 'none';
            }
            
            showSuccess(message) {
                const alert = document.createElement('div');
                alert.className = 'alert alert-success';
                alert.textContent = message;
                this.processBtn.parentNode.insertBefore(alert, this.processBtn.nextSibling);
                setTimeout(() => {
                    if (alert.parentNode) {
                        alert.parentNode.removeChild(alert);
                    }
                }, 3000);
            }
        }
        
        document.addEventListener('DOMContentLoaded', () => {
            new AdvancedPDFExtractor();
        });
    </script>
</body>
</html>"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/extract', methods=['POST'])
def extract_api():
    try:
        if 'pdf' not in request.files:
            return jsonify({'error': 'No PDF uploaded'}), 400
            
        file = request.files['pdf']
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Please upload PDF'}), 400
            
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            file.save(temp_pdf.name)
            temp_pdf_path = temp_pdf.name
            
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_json:
            temp_json_path = temp_json.name
            
        try:
            # Call the advanced extract function from main.py
            print(f"🔍 Processing {file.filename} with advanced algorithms...")
            success = extract_pdf_outline(temp_pdf_path, temp_json_path)
            if not success:
                return jsonify({'error': 'Advanced processing failed'}), 500
                
            # Read results with all the advanced metadata
            with open(temp_json_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
                
            print(f"✅ Advanced processing complete: {len(results.get('outline', []))} headings detected")
            if 'metadata' in results:
                print(f"📊 Language: {results['metadata'].get('detected_language', 'Unknown')}")
                print(f"📊 Processing time: {results['metadata'].get('processing_time_seconds', 'N/A')}s")
                
            return jsonify(results)
            
        finally:
            # Clean up temporary files
            try:
                os.unlink(temp_pdf_path)
                os.unlink(temp_json_path)
            except:
                pass
                
    except Exception as e:
        print(f"❌ Advanced processing error: {str(e)}")
        return jsonify({'error': f'Advanced processing error: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'version': 'advanced',
        'features': [
            'multi_language_detection',
            'noise_filtering', 
            'improved_heading_detection',
            'text_cleaning',
            'statistical_analysis'
        ]
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting Advanced PDF Extractor on port {port}")
    print(f"🌐 Access at: http://localhost:{port}")
    print(f"✨ Features: Language detection, noise filtering, AI heading detection")
    app.run(host='0.0.0.0', port=port, debug=False)