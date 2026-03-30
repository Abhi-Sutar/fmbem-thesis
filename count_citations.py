import re
import sys

def count_distinct_citations(bcf_file):
    with open(bcf_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Regex to capture content inside <bcf:citekey> tags
    pattern = re.compile(r'<bcf:citekey[^>]*>(.*?)</bcf:citekey>', re.DOTALL)
    keys = pattern.findall(content)
    
    # Remove duplicates and count
    unique_keys = set(keys)
    return len(unique_keys)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python count_citations.py <main.bcf>")
        sys.exit(1)
    
    count = count_distinct_citations(sys.argv[1])
    print(f"Number of distinct cited references: {count}")