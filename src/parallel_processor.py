"""
Parallel Processor - Multi-threaded page processing for improved performance
Uses ThreadPoolExecutor for CPU-bound tasks while maintaining thread safety
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional
import threading
from pathlib import Path
import fitz  # PyMuPDF

class ParallelPDFProcessor:
    """Parallel processing system for PDF analysis"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.logger = logging.getLogger(__name__)
        
        # Auto-detect optimal worker count
        if max_workers is None:
            import os
            # Use 75% of available cores, minimum 2, maximum 8
            cpu_count = os.cpu_count() or 4
            max_workers = max(2, min(8, int(cpu_count * 0.75)))
        
        self.max_workers = max_workers
        self.processing_stats = {
            'pages_processed': 0,
            'total_processing_time': 0.0,
            'parallel_efficiency': 0.0,
            'worker_utilization': {}
        }
        
        self.logger.info(f"Initialized parallel processor with {self.max_workers} workers")
    
    def process_pdf_parallel(self, pdf_path: str, 
                           processing_function: Callable,
                           chunk_size: int = 5) -> List[Dict[str, Any]]:
        """
        Process PDF pages in parallel
        
        Args:
            pdf_path: Path to PDF file
            processing_function: Function to apply to each page
            chunk_size: Number of pages per processing chunk
            
        Returns:
            List of processed results from all pages
        """
        start_time = time.time()
        
        try:
            # Open PDF and get page count
            doc = fitz.open(pdf_path)
            total_pages = min(doc.page_count, 50)  # Limit to 50 pages
            doc.close()
            
            if total_pages == 0:
                self.logger.warning("PDF has no pages")
                return []
            
            self.logger.info(f"Processing {total_pages} pages in parallel with {self.max_workers} workers")
            
            # Create page chunks for parallel processing
            page_chunks = self._create_page_chunks(total_pages, chunk_size)
            
            # Process chunks in parallel
            all_results = []
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all chunks for processing
                future_to_chunk = {
                    executor.submit(self._process_page_chunk, pdf_path, chunk, processing_function): chunk
                    for chunk in page_chunks
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    try:
                        chunk_results = future.result()
                        all_results.extend(chunk_results)
                        
                        self.logger.debug(f"Completed chunk {chunk['start_page']}-{chunk['end_page']} "
                                        f"with {len(chunk_results)} blocks")
                        
                    except Exception as e:
                        self.logger.error(f"Chunk {chunk} processing failed: {e}")
            
            # Sort results by page order
            all_results.sort(key=lambda x: (x.get('page', 0), x.get('bbox', [0, 0, 0, 0])[1]))
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_processing_stats(total_pages, processing_time)
            
            self.logger.info(f"Parallel processing completed: {len(all_results)} blocks from "
                           f"{total_pages} pages in {processing_time:.2f}s")
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"Parallel processing failed: {e}")
            # Fallback to sequential processing
            return self._fallback_sequential_processing(pdf_path, processing_function)
    
    def _create_page_chunks(self, total_pages: int, chunk_size: int) -> List[Dict[str, Any]]:
        """Create page chunks for parallel processing"""
        
        chunks = []
        
        for start_page in range(1, total_pages + 1, chunk_size):
            end_page = min(start_page + chunk_size - 1, total_pages)
            
            chunk = {
                'start_page': start_page,
                'end_page': end_page,
                'page_count': end_page - start_page + 1,
                'chunk_id': len(chunks)
            }
            
            chunks.append(chunk)
        
        self.logger.debug(f"Created {len(chunks)} chunks for {total_pages} pages")
        return chunks
    
    def _process_page_chunk(self, pdf_path: str, chunk: Dict[str, Any], 
                          processing_function: Callable) -> List[Dict[str, Any]]:
        """Process a single chunk of pages"""
        
        thread_id = threading.current_thread().ident
        chunk_start_time = time.time()
        
        try:
            # Open PDF document (each thread needs its own instance)
            doc = fitz.open(pdf_path)
            
            chunk_results = []
            
            # Process pages in this chunk
            for page_num in range(chunk['start_page'], chunk['end_page'] + 1):
                try:
                    page_index = page_num - 1  # fitz uses 0-based indexing
                    
                    if page_index < doc.page_count:
                        page = doc[page_index]
                        
                        # Apply processing function to page
                        page_results = processing_function(page, page_num)
                        
                        if page_results:
                            chunk_results.extend(page_results)
                        
                        # Memory cleanup for large documents
                        if page_num % 10 == 0:
                            import gc
                            gc.collect()
                
                except Exception as e:
                    self.logger.error(f"Error processing page {page_num} in thread {thread_id}: {e}")
                    continue
            
            doc.close()
            
            # Update worker utilization stats
            processing_time = time.time() - chunk_start_time
            self._update_worker_stats(thread_id, chunk['page_count'], processing_time)
            
            return chunk_results
            
        except Exception as e:
            self.logger.error(f"Chunk processing failed in thread {thread_id}: {e}")
            return []
    
    def _update_worker_stats(self, thread_id: int, pages_processed: int, processing_time: float):
        """Update worker utilization statistics"""
        
        if thread_id not in self.processing_stats['worker_utilization']:
            self.processing_stats['worker_utilization'][thread_id] = {
                'pages_processed': 0,
                'total_time': 0.0,
                'chunks_completed': 0
            }
        
        worker_stats = self.processing_stats['worker_utilization'][thread_id]
        worker_stats['pages_processed'] += pages_processed
        worker_stats['total_time'] += processing_time
        worker_stats['chunks_completed'] += 1
    
    def _update_processing_stats(self, total_pages: int, total_time: float):
        """Update overall processing statistics"""
        
        self.processing_stats['pages_processed'] = total_pages
        self.processing_stats['total_processing_time'] = total_time
        
        # Calculate theoretical sequential time (rough estimate)
        avg_time_per_page = total_time / max(total_pages, 1)
        theoretical_sequential_time = avg_time_per_page * total_pages
        
        # Calculate parallel efficiency
        if theoretical_sequential_time > 0:
            self.processing_stats['parallel_efficiency'] = theoretical_sequential_time / total_time
        else:
            self.processing_stats['parallel_efficiency'] = 1.0
    
    def _fallback_sequential_processing(self, pdf_path: str, 
                                      processing_function: Callable) -> List[Dict[str, Any]]:
        """Fallback to sequential processing if parallel fails"""
        
        self.logger.warning("Falling back to sequential processing")
        
        try:
            doc = fitz.open(pdf_path)
            total_pages = min(doc.page_count, 50)
            
            all_results = []
            
            for page_num in range(1, total_pages + 1):
                try:
                    page_index = page_num - 1
                    page = doc[page_index]
                    
                    page_results = processing_function(page, page_num)
                    
                    if page_results:
                        all_results.extend(page_results)
                        
                except Exception as e:
                    self.logger.error(f"Error processing page {page_num}: {e}")
                    continue
            
            doc.close()
            return all_results
            
        except Exception as e:
            self.logger.error(f"Sequential fallback also failed: {e}")
            return []
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get detailed processing statistics"""
        
        stats = self.processing_stats.copy()
        
        # Add derived metrics
        if stats['total_processing_time'] > 0:
            stats['pages_per_second'] = stats['pages_processed'] / stats['total_processing_time']
        else:
            stats['pages_per_second'] = 0
        
        # Worker efficiency analysis
        worker_stats = stats['worker_utilization']
        if worker_stats:
            stats['active_workers'] = len(worker_stats)
            stats['avg_pages_per_worker'] = stats['pages_processed'] / len(worker_stats)
            
            # Calculate load balancing efficiency
            worker_page_counts = [w['pages_processed'] for w in worker_stats.values()]
            if worker_page_counts:
                max_pages = max(worker_page_counts)
                min_pages = min(worker_page_counts)
                stats['load_balance_ratio'] = min_pages / max_pages if max_pages > 0 else 1.0
            else:
                stats['load_balance_ratio'] = 1.0
        
        return stats
    
    def create_performance_report(self, output_path: str = "/app/output/parallel_performance.txt"):
        """Create detailed performance report"""
        
        stats = self.get_processing_stats()
        
        lines = [
            "PARALLEL PROCESSING PERFORMANCE REPORT",
            "=" * 50,
            "",
            f"Configuration:",
            f"  Max Workers: {self.max_workers}",
            f"  Pages Processed: {stats['pages_processed']}",
            "",
            f"Performance Metrics:",
            f"  Total Processing Time: {stats['total_processing_time']:.3f}s",
            f"  Pages per Second: {stats['pages_per_second']:.2f}",
            f"  Parallel Efficiency: {stats['parallel_efficiency']:.2f}x",
            ""
        ]
        
        # Worker utilization details
        if stats['worker_utilization']:
            lines.append("Worker Utilization:")
            lines.append("-" * 25)
            
            for thread_id, worker_stats in stats['worker_utilization'].items():
                avg_time_per_page = worker_stats['total_time'] / max(worker_stats['pages_processed'], 1)
                
                lines.append(f"  Worker {thread_id}:")
                lines.append(f"    Pages: {worker_stats['pages_processed']}")
                lines.append(f"    Time: {worker_stats['total_time']:.3f}s")
                lines.append(f"    Chunks: {worker_stats['chunks_completed']}")
                lines.append(f"    Avg Time/Page: {avg_time_per_page:.3f}s")
                lines.append("")
            
            lines.append(f"Load Balance Ratio: {stats['load_balance_ratio']:.3f}")
            lines.append("")
        
        # Recommendations
        lines.append("Performance Analysis:")
        lines.append("-" * 25)
        
        if stats['parallel_efficiency'] >= 1.5:
            lines.append("✓ Excellent parallel performance - well optimized")
        elif stats['parallel_efficiency'] >= 1.2:
            lines.append("✓ Good parallel performance - minor optimization possible")
        elif stats['parallel_efficiency'] >= 1.0:
            lines.append("⚠ Moderate parallel performance - consider optimization")
        else:
            lines.append("⚠ Poor parallel performance - sequential might be better")
        
        if stats.get('load_balance_ratio', 1) < 0.8:
            lines.append("⚠ Uneven load balancing detected")
        
        lines.append("")
        lines.append("=" * 50)
        
        # Write report
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            self.logger.info(f"Performance report saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save performance report: {e}")
    
    @staticmethod
    def create_page_processing_function(extraction_method: str = 'comprehensive') -> Callable:
        """
        Create a page processing function for parallel execution
        
        Args:
            extraction_method: Type of extraction to perform
            
        Returns:
            Function that can process a single page
        """
        
        def process_page_comprehensive(page, page_num: int) -> List[Dict[str, Any]]:
            """Comprehensive page processing with all metadata"""
            
            blocks = []
            
            try:
                # Get text blocks with font information
                text_dict = page.get_text("dict")
                
                for block in text_dict.get("blocks", []):
                    if "lines" not in block:
                        continue
                    
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span.get("text", "").strip()
                            if not text or len(text) < 2:
                                continue
                            
                            # Extract comprehensive font metadata
                            font_info = {
                                'size': float(span.get("size", 12.0)),
                                'flags': int(span.get("flags", 0)),
                                'name': span.get("font", "").lower(),
                                'is_bold': bool(span.get("flags", 0) & 2**4),
                                'is_italic': bool(span.get("flags", 0) & 2**1),
                                'color': span.get("color", 0)
                            }
                            
                            # Position information
                            bbox = span.get("bbox", [0, 0, 0, 0])
                            
                            block_data = {
                                'text': text,
                                'page': page_num,
                                'bbox': bbox,
                                'font_info': font_info,
                                'width': bbox[2] - bbox[0],
                                'height': bbox[3] - bbox[1],
                                'line_bbox': line.get("bbox", bbox),
                                'block_bbox': block.get("bbox", bbox),
                                'processing_method': 'parallel_comprehensive'
                            }
                            
                            blocks.append(block_data)
                            
            except Exception as e:
                # Log error but don't stop processing
                logging.getLogger(__name__).error(f"Error processing page {page_num}: {e}")
            
            return blocks
        
        def process_page_fast(page, page_num: int) -> List[Dict[str, Any]]:
            """Fast page processing with minimal metadata"""
            
            blocks = []
            
            try:
                # Simple text extraction
                text_dict = page.get_text("dict")
                
                for block in text_dict.get("blocks", []):
                    if "lines" not in block:
                        continue
                    
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span.get("text", "").strip()
                            if not text:
                                continue
                            
                            bbox = span.get("bbox", [0, 0, 0, 0])
                            
                            block_data = {
                                'text': text,
                                'page': page_num,
                                'bbox': bbox,
                                'font_info': {
                                    'size': float(span.get("size", 12.0)),
                                    'is_bold': bool(span.get("flags", 0) & 2**4)
                                },
                                'processing_method': 'parallel_fast'
                            }
                            
                            blocks.append(block_data)
                            
            except Exception as e:
                logging.getLogger(__name__).error(f"Error in fast processing page {page_num}: {e}")
            
            return blocks
        
        # Return appropriate function based on method
        if extraction_method == 'fast':
            return process_page_fast
        else:
            return process_page_comprehensive