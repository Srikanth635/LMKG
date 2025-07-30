"""
Command Line Interface for SOMA Object Analysis
"""

from .main import cli, main

__all__ = [
    "cli",
    "main"
]

# Entry point for console script
def console_main():
    """Entry point for the soma-analyze console command"""
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        exit(1)
    except Exception as e:
        print(f"❌ CLI Error: {e}")
        exit(1)