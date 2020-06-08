""" 
Define the argparser for converting from Supervisely to MS-COCO
"""
import argparse

__all__ = ['argument_parser']

def argument_parser(epilog: str = None)-> argparse.ArgumentParser:
    """
    Create an argument parser for initiating the conversion process.

    Args:
        epilog (str): epilog passed to ArgumentParser describing usage
    
    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(epilog= epilog or f"""
    Example:
        python supervisely2coco.py --meta /path/to/meta.json --annotations /path/to/annotations/folder --output /path/to/output.json
    """)

    parser.add_argument("--meta", "-m", help = 'Path to meta.json file')
    parser.add_argument("--annotations", "-a", help = "Annotations base dir")
    parser.add_argument("--output", "-o", help = "Output json filename")
    parser.add_argument("-image-name", '-n', action = 'store_true',
                        help = 'Save only filename(without absolute path')


    return parser

def test():
    pass