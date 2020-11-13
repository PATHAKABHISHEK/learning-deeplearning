import pandas as pd

# OR Perceptron

weight1 = 1.0
weight2 = 1.0
bias = -1.0

test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
test_outputs = [False, True, True, True]
outputs = []

for test_input, test_output in zip(test_inputs, test_outputs):
    linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
    output = (linear_combination >=0)
    is_correct_string = 'Yes' if output == test_output else 'No'
    outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])

num_wrong = len([output[4] for output in outputs if output[4] == 'No'])

output_frame = pd.DataFrame(outputs, columns=['Input 1', ' Input 2', ' Linear Combination', ' Activation Output', ' Is Correct' ])

if not num_wrong:
    print("You got All Correct\n")
else:
    print("You got {} wrong\n".format(num_wrong))

print(output_frame.to_string(index=False))
