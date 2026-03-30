import re
import sys

with open(sys.argv[1], 'r') as f:
    content = f.read()

# Find all citation keys in \citation{...} lines
keys = re.findall(r'\\citation\{([^}]*)\}', content)

# Remove duplicates and count
unique_keys = set(keys)
print(f"Number of distinct cited references: {len(unique_keys)}")