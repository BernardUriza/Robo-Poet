"""
Async I/O Optimizer

High-performance asynchronous I/O operations for file handling.
Addresses blocking I/O bottlenecks identified in the performance analysis.
"""

import asyncio
import aiofiles
from pathlib import Path
from typing import List, Dict, Optional, Iterator, Tuple, AsyncIterator
import logging
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor
import json


@dataclass
class AsyncIOConfig:
    """Configuration for async I/O operations."""
    max_concurrent_files: int = 10
    chunk_size: int = 64 * 1024  # 64KB chunks
    buffer_size: int = 8 * 1024 * 1024  # 8MB buffer
    timeout_seconds: float = 30.0
    enable_compression: bool = False
    use_thread_pool: bool = True
    max_workers: int = 4


class AsyncFileProcessor:
    """
    Asynchronous file processor for high-performance I/O operations.
    
    Key features:
    - Non-blocking file operations
    - Concurrent file processing
    - Memory-efficient streaming
    - Progress monitoring
    - Error handling with retry logic
    """
    
    def __init__(self, config: AsyncIOConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.semaphore = asyncio.Semaphore(config.max_concurrent_files)
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_workers) if config.use_thread_pool else None
        
        # Performance metrics
        self.stats = {
            'files_processed': 0,
            'bytes_read': 0,
            'bytes_written': 0,
            'processing_time': 0.0,
            'errors': 0
        }
    
    async def read_file_async(self, filepath: str) -> str:
        """
        Asynchronously read entire file content.
        
        Args:
            filepath: Path to file
            
        Returns:
            File content as string
        """
        async with self.semaphore:
            try:
                start_time = time.time()
                
                async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
                    content = await f.read()
                
                # Update stats
                self.stats['files_processed'] += 1
                self.stats['bytes_read'] += len(content.encode('utf-8'))
                self.stats['processing_time'] += time.time() - start_time
                
                return content
                
            except Exception as e:
                self.stats['errors'] += 1
                self.logger.error(f"Error reading file {filepath}: {e}")
                raise
    
    async def read_file_chunks_async(self, filepath: str) -> AsyncIterator[str]:
        """
        Asynchronously read file in chunks for memory efficiency.
        
        Args:
            filepath: Path to file
            
        Yields:
            File content chunks
        """
        async with self.semaphore:
            try:
                async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
                    while True:
                        chunk = await f.read(self.config.chunk_size)
                        if not chunk:
                            break
                        
                        self.stats['bytes_read'] += len(chunk.encode('utf-8'))
                        yield chunk
                        
            except Exception as e:
                self.stats['errors'] += 1
                self.logger.error(f"Error reading file chunks {filepath}: {e}")
                raise
    
    async def write_file_async(self, filepath: str, content: str) -> bool:
        """
        Asynchronously write content to file.
        
        Args:
            filepath: Destination file path
            content: Content to write
            
        Returns:
            True if successful
        """
        async with self.semaphore:
            try:
                start_time = time.time()
                
                # Ensure parent directory exists
                Path(filepath).parent.mkdir(parents=True, exist_ok=True)
                
                async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
                    await f.write(content)
                
                # Update stats
                self.stats['bytes_written'] += len(content.encode('utf-8'))
                self.stats['processing_time'] += time.time() - start_time
                
                return True
                
            except Exception as e:
                self.stats['errors'] += 1
                self.logger.error(f"Error writing file {filepath}: {e}")
                raise
    
    async def process_files_concurrent(self, file_paths: List[str], 
                                     processor_func: callable) -> List[Tuple[str, any]]:
        """
        Process multiple files concurrently.
        
        Args:
            file_paths: List of file paths to process
            processor_func: Function to process each file content
            
        Returns:
            List of (filepath, result) tuples
        """
        async def process_single_file(filepath: str) -> Tuple[str, any]:
            try:
                content = await self.read_file_async(filepath)
                if asyncio.iscoroutinefunction(processor_func):
                    result = await processor_func(content)
                else:
                    # Run CPU-intensive processing in thread pool
                    if self.thread_pool:
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(self.thread_pool, processor_func, content)
                    else:
                        result = processor_func(content)
                
                return filepath, result
                
            except Exception as e:
                self.logger.error(f"Error processing file {filepath}: {e}")
                return filepath, None
        
        # Process files concurrently with semaphore limiting
        tasks = [process_single_file(filepath) for filepath in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [
            result for result in results 
            if not isinstance(result, Exception)
        ]
        
        self.logger.info(f"Processed {len(valid_results)}/{len(file_paths)} files successfully")
        
        return valid_results
    
    async def stream_large_file(self, filepath: str, chunk_processor: callable) -> Dict[str, any]:
        """
        Stream process large file without loading into memory.
        
        Args:
            filepath: Path to large file
            chunk_processor: Function to process each chunk
            
        Returns:
            Processing results and statistics
        """
        results = []
        total_chunks = 0
        total_size = 0
        start_time = time.time()
        
        try:
            async for chunk in self.read_file_chunks_async(filepath):
                if asyncio.iscoroutinefunction(chunk_processor):
                    result = await chunk_processor(chunk)
                else:
                    # Run CPU-intensive processing in thread pool
                    if self.thread_pool:
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(self.thread_pool, chunk_processor, chunk)
                    else:
                        result = chunk_processor(chunk)
                
                if result is not None:
                    results.append(result)
                
                total_chunks += 1
                total_size += len(chunk.encode('utf-8'))
                
                # Log progress for very large files
                if total_chunks % 100 == 0:
                    self.logger.info(f"Processed {total_chunks} chunks, {total_size / (1024*1024):.1f}MB")
        
        except Exception as e:
            self.logger.error(f"Error streaming file {filepath}: {e}")
            raise
        
        processing_time = time.time() - start_time
        
        return {
            'results': results,
            'chunks_processed': total_chunks,
            'bytes_processed': total_size,
            'processing_time': processing_time,
            'throughput_mbps': (total_size / (1024*1024)) / processing_time if processing_time > 0 else 0
        }
    
    async def batch_write_files(self, file_data: Dict[str, str]) -> Dict[str, bool]:
        """
        Write multiple files concurrently.
        
        Args:
            file_data: Dictionary of {filepath: content}
            
        Returns:
            Dictionary of {filepath: success_status}
        """
        async def write_single_file(filepath: str, content: str) -> Tuple[str, bool]:
            try:
                success = await self.write_file_async(filepath, content)
                return filepath, success
            except Exception as e:
                self.logger.error(f"Failed to write {filepath}: {e}")
                return filepath, False
        
        # Create tasks for concurrent writing
        tasks = [
            write_single_file(filepath, content) 
            for filepath, content in file_data.items()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert to dictionary
        result_dict = {}
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Write task failed: {result}")
            else:
                filepath, success = result
                result_dict[filepath] = success
        
        return result_dict
    
    async def find_files_async(self, directory: str, pattern: str = "*", 
                              max_depth: int = 10) -> List[str]:
        """
        Asynchronously find files matching pattern.
        
        Args:
            directory: Directory to search
            pattern: Glob pattern to match
            max_depth: Maximum recursion depth
            
        Returns:
            List of matching file paths
        """
        def find_files_sync(dir_path: Path, pattern: str, max_depth: int) -> List[str]:
            """Synchronous file finding to run in thread pool."""
            try:
                if max_depth <= 0:
                    return []
                
                files = []
                for item in dir_path.iterdir():
                    if item.is_file() and item.match(pattern):
                        files.append(str(item))
                    elif item.is_dir() and max_depth > 1:
                        files.extend(find_files_sync(item, pattern, max_depth - 1))
                
                return files
                
            except (PermissionError, OSError) as e:
                self.logger.warning(f"Cannot access directory {dir_path}: {e}")
                return []
        
        # Run file finding in thread pool to avoid blocking
        if self.thread_pool:
            loop = asyncio.get_event_loop()
            file_paths = await loop.run_in_executor(
                self.thread_pool, 
                find_files_sync, 
                Path(directory), 
                pattern, 
                max_depth
            )
        else:
            file_paths = find_files_sync(Path(directory), pattern, max_depth)
        
        return file_paths
    
    def get_performance_stats(self) -> Dict[str, any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        stats = self.stats.copy()
        
        if stats['processing_time'] > 0:
            stats['throughput_files_per_sec'] = stats['files_processed'] / stats['processing_time']
            stats['throughput_mbps'] = (stats['bytes_read'] / (1024*1024)) / stats['processing_time']
        else:
            stats['throughput_files_per_sec'] = 0
            stats['throughput_mbps'] = 0
        
        stats['total_bytes'] = stats['bytes_read'] + stats['bytes_written']
        stats['error_rate'] = stats['errors'] / max(1, stats['files_processed'])
        
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.stats = {
            'files_processed': 0,
            'bytes_read': 0,
            'bytes_written': 0,
            'processing_time': 0.0,
            'errors': 0
        }
    
    async def close(self):
        """Clean up resources."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)


class AsyncTextCorpusLoader:
    """
    Specialized async loader for text corpus files.
    
    Optimized for ML training data loading with validation and preprocessing.
    """
    
    def __init__(self, config: AsyncIOConfig):
        self.async_processor = AsyncFileProcessor(config)
        self.logger = logging.getLogger(__name__)
    
    async def load_corpus_files(self, file_paths: List[str], 
                               min_size_kb: int = 1) -> Dict[str, str]:
        """
        Load and validate multiple corpus files asynchronously.
        
        Args:
            file_paths: List of corpus file paths
            min_size_kb: Minimum file size in KB
            
        Returns:
            Dictionary of {filepath: content} for valid files
        """
        async def validate_and_load_file(filepath: str) -> Tuple[str, Optional[str]]:
            try:
                # Check file size first
                file_size = Path(filepath).stat().st_size
                if file_size < min_size_kb * 1024:
                    return filepath, None
                
                # Load content
                content = await self.async_processor.read_file_async(filepath)
                
                # Basic validation
                if len(content.strip()) < 100:  # Minimum meaningful content
                    return filepath, None
                
                return filepath, content
                
            except Exception as e:
                self.logger.warning(f"Failed to load corpus file {filepath}: {e}")
                return filepath, None
        
        # Process all files concurrently
        tasks = [validate_and_load_file(fp) for fp in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter valid results
        valid_corpus = {}
        for result in results:
            if isinstance(result, Exception):
                continue
            
            filepath, content = result
            if content is not None:
                valid_corpus[filepath] = content
        
        self.logger.info(f"Loaded {len(valid_corpus)}/{len(file_paths)} corpus files")
        return valid_corpus
    
    async def save_processed_corpus(self, processed_data: Dict[str, any], 
                                   output_dir: str) -> bool:
        """
        Save processed corpus data asynchronously.
        
        Args:
            processed_data: Dictionary of processed data to save
            output_dir: Output directory
            
        Returns:
            True if successful
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare files to write
        files_to_write = {}
        
        for key, data in processed_data.items():
            if isinstance(data, str):
                # Text data
                files_to_write[str(output_path / f"{key}.txt")] = data
            else:
                # JSON serializable data
                files_to_write[str(output_path / f"{key}.json")] = json.dumps(data, indent=2)
        
        # Write all files concurrently
        results = await self.async_processor.batch_write_files(files_to_write)
        
        success_count = sum(1 for success in results.values() if success)
        total_count = len(results)
        
        self.logger.info(f"Saved {success_count}/{total_count} processed corpus files")
        
        return success_count == total_count


# Utility functions for easy async operations
async def read_text_files_async(file_paths: List[str], 
                               config: Optional[AsyncIOConfig] = None) -> Dict[str, str]:
    """
    Convenience function to read multiple text files asynchronously.
    
    Args:
        file_paths: List of file paths to read
        config: Optional configuration
        
    Returns:
        Dictionary of {filepath: content}
    """
    if config is None:
        config = AsyncIOConfig()
    
    processor = AsyncFileProcessor(config)
    
    try:
        results = await processor.process_files_concurrent(
            file_paths, 
            lambda content: content  # Identity function - just return content
        )
        
        return {filepath: content for filepath, content in results if content is not None}
    
    finally:
        await processor.close()


async def write_text_files_async(file_data: Dict[str, str], 
                                config: Optional[AsyncIOConfig] = None) -> Dict[str, bool]:
    """
    Convenience function to write multiple text files asynchronously.
    
    Args:
        file_data: Dictionary of {filepath: content}
        config: Optional configuration
        
    Returns:
        Dictionary of {filepath: success_status}
    """
    if config is None:
        config = AsyncIOConfig()
    
    processor = AsyncFileProcessor(config)
    
    try:
        return await processor.batch_write_files(file_data)
    
    finally:
        await processor.close()


# Example usage and performance test
if __name__ == "__main__":
    async def main():
        # Configuration for high performance
        config = AsyncIOConfig(
            max_concurrent_files=20,
            chunk_size=128 * 1024,  # 128KB chunks
            use_thread_pool=True,
            max_workers=4
        )
        
        processor = AsyncFileProcessor(config)
        
        # Example: Process multiple files
        file_paths = ["test1.txt", "test2.txt", "test3.txt"]
        
        def simple_processor(content: str) -> int:
            return len(content.split())
        
        try:
            # Process files concurrently
            results = await processor.process_files_concurrent(file_paths, simple_processor)
            
            print("Processing results:")
            for filepath, word_count in results:
                print(f"  {filepath}: {word_count} words")
            
            # Show performance stats
            stats = processor.get_performance_stats()
            print("\\nPerformance Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        
        finally:
            await processor.close()
    
    # Run example
    # asyncio.run(main())