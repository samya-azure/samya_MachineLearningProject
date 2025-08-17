
# OR gate in Python
def OR_gate(a, b):
    return a | b  # bitwise OR

# Test all input combinations
inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

print("A B | Output")
print("------------")
for a, b in inputs:
    result = OR_gate(a, b)
    print(f"{a} {b} |   {result}")
