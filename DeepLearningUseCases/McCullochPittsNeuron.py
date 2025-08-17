
# ----- Door Access Control system using McCulloch & Pitts Neuron ------
# Business Logic:
    # Doors open only if,
        # 1. Has ID Card = 1 , (means the person has ID card)
        # 2. Knows PIN = 1, (means, the person knows the correct PIN)
        # 3. Not blacklisted = 1, (means, the person is not blacklisted)
    # so, all must be true (1), i.e., similar to AND logic gate

def mcp_neuron(inputs, weights, threshold):
    total = sum(i * w for i, w in zip(inputs, weights))
    return 1 if total >= threshold else 0

print("🔐 Door Access Control System")
print("Format: [Has ID, Knows PIN, Not Blacklisted] => Access")

# Try all possible combinations of inputs
for id_card in [0, 1]:
    for knows_pin in [0, 1]:
        for not_blacklisted in [0, 1]:
            inputs = [id_card, knows_pin, not_blacklisted]
            weights = [1, 1, 1]
            threshold = 3   # All 3 conditions must be true (1+1+1)
            output = mcp_neuron(inputs, weights, threshold)
            access = "✅ Access GRANTED" if output == 1 else "❌ Access DENIED"
            print(f"Inputs: {inputs} => {access}")
