
# XOR gate in Python
def XOR_gate(a, b):
    return a ^ b  # bitwise XOR

# Test XOR for all inputs
inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

print("A B | Output")
print("------------")
for a, b in inputs:
    result = XOR_gate(a, b)
    print(f"{a} {b} |   {result}")
