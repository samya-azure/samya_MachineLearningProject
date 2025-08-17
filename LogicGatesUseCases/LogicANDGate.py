
# Logic AND gate in Python
def AND_gate(a, b):
    return a & b  # bitwise AND

# Test the AND gate
inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

for a, b in inputs:
    result = AND_gate(a, b)
    print(f"{a} AND {b} = {result}")
