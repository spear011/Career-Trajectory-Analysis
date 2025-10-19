"""
Performance benchmarking utilities
"""
import time
import psutil
import os
from contextlib import contextmanager
from functools import wraps

class Timer:
    """Simple timer context manager"""
    
    def __init__(self, name="Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        print(f"⏱️  {self.name}: {self.elapsed:.2f}s")


def timer_decorator(func):
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"⏱️  {func.__name__}: {elapsed:.2f}s")
        return result
    return wrapper


class MemoryMonitor:
    """Monitor memory usage"""
    
    def __init__(self, name="Operation"):
        self.name = name
        self.process = psutil.Process(os.getpid())
        self.start_memory = None
        self.peak_memory = None
    
    def __enter__(self):
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        return self
    
    def __exit__(self, *args):
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - self.start_memory
        print(f"💾 {self.name}: +{memory_increase:.1f} MB (Current: {current_memory:.1f} MB)")


@contextmanager
def performance_monitor(name="Operation"):
    """Monitor both time and memory"""
    print(f"\n{'='*60}")
    print(f"📊 {name}")
    print(f"{'='*60}")
    
    process = psutil.Process(os.getpid())
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    yield
    
    elapsed = time.time() - start_time
    current_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = current_memory - start_memory
    
    print(f"⏱️  Time: {elapsed:.2f}s")
    print(f"💾 Memory: +{memory_increase:.1f} MB (Current: {current_memory:.1f} MB)")
    print(f"{'='*60}\n")


class PipelineBenchmark:
    """Comprehensive pipeline benchmark"""
    
    def __init__(self):
        self.timings = {}
        self.memory_usage = {}
        self.process = psutil.Process(os.getpid())
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024
    
    def start_stage(self, stage_name):
        """Start timing a stage"""
        self.timings[stage_name] = {'start': time.time()}
        self.memory_usage[stage_name] = {
            'start': self.process.memory_info().rss / 1024 / 1024
        }
    
    def end_stage(self, stage_name):
        """End timing a stage"""
        if stage_name not in self.timings:
            print(f"Warning: Stage '{stage_name}' was not started")
            return
        
        self.timings[stage_name]['end'] = time.time()
        self.timings[stage_name]['elapsed'] = (
            self.timings[stage_name]['end'] - self.timings[stage_name]['start']
        )
        
        current_memory = self.process.memory_info().rss / 1024 / 1024
        self.memory_usage[stage_name]['end'] = current_memory
        self.memory_usage[stage_name]['delta'] = (
            current_memory - self.memory_usage[stage_name]['start']
        )
    
    def print_report(self):
        """Print comprehensive benchmark report"""
        total_time = time.time() - self.start_time
        total_memory = self.process.memory_info().rss / 1024 / 1024 - self.start_memory
        
        print("\n" + "="*80)
        print("📊 PIPELINE PERFORMANCE REPORT")
        print("="*80)
        
        print("\n⏱️  TIMING BREAKDOWN:")
        print("-" * 80)
        print(f"{'Stage':<40} {'Time (s)':<15} {'% of Total':<15}")
        print("-" * 80)
        
        for stage, timing in self.timings.items():
            if 'elapsed' in timing:
                percentage = (timing['elapsed'] / total_time) * 100
                print(f"{stage:<40} {timing['elapsed']:>10.2f}     {percentage:>10.1f}%")
        
        print("-" * 80)
        print(f"{'TOTAL':<40} {total_time:>10.2f}     {100:>10.1f}%")
        
        print("\n💾 MEMORY USAGE:")
        print("-" * 80)
        print(f"{'Stage':<40} {'Delta (MB)':<15}")
        print("-" * 80)
        
        for stage, memory in self.memory_usage.items():
            if 'delta' in memory:
                print(f"{stage:<40} {memory['delta']:>10.1f}")
        
        print("-" * 80)
        print(f"{'TOTAL':<40} {total_memory:>10.1f}")
        print("="*80 + "\n")
    
    def save_report(self, filepath):
        """Save report to file"""
        import json
        
        report = {
            'total_time': time.time() - self.start_time,
            'total_memory_mb': self.process.memory_info().rss / 1024 / 1024 - self.start_memory,
            'stages': {}
        }
        
        for stage in self.timings:
            report['stages'][stage] = {
                'time_seconds': self.timings[stage].get('elapsed', 0),
                'memory_mb': self.memory_usage[stage].get('delta', 0)
            }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Benchmark report saved to: {filepath}")