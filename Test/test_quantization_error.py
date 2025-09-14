import torch
from helper import linear_q_with_scale_and_zero_point, linear_dequantization

def get_q_scale_and_zero_point(tensor, dtype=torch.int8):
    q_min, q_max = torch.iinfo(dtype).min, torch.iinfo(dtype).max
    r_min, r_max = tensor.min().item(), tensor.max().item()
    scale = (r_max - r_min) / (q_max - q_min)
    zero_point = q_min - (r_min / scale)
    if zero_point < q_min:
        zero_point = q_min
    elif zero_point > q_max:
        zero_point = q_max
    else:
        zero_point = int(round(zero_point))
    return scale, zero_point

def test_quantization_error(tensor):
    scale, zero_point = get_q_scale_and_zero_point(tensor)
    print(f"Scale: {scale}, Zero Point: {zero_point}")
    quantized_tensor = linear_q_with_scale_and_zero_point(tensor, scale, zero_point)
    dequantized_tensor = linear_dequantization(quantized_tensor, scale, zero_point)
    print("Original Tensor\n:", tensor)
    print(f"Quantized Tensor:\n{quantized_tensor}")
    print(f"Dequantized Tensor:\n{dequantized_tensor}")
    error = (dequantized_tensor - tensor).square().mean()
    return error.item()

# Test Case 1: Large dynamic range
print("Test Case1:")
tensor1 = torch.tensor([[191.6, -13.5, 728.6],
                        [92.14, 295.5, -184],
                        [0, 684.6, 245.5]])
error1 = test_quantization_error(tensor1)
print("\n")

# Test Case 2: Small dynamic range
print("Test Case2:")
tensor2 = torch.tensor([[0.1, 0.2, 0.3],
                        [0.4, 0.5, 0.6],
                        [0.7, 0.8, 0.9]])
error2 = test_quantization_error(tensor2)
print("\n")

# Test Case 3: Non-uniform distribution
print("Test Case3:")
tensor3 = torch.tensor([[1.0, 1.1, 1.2],
                        [1.3, 1.4, 1.5],
                        [10.0, 10.1, 10.2]])  # Non-uniform distribution
error3 = test_quantization_error(tensor3)
print("\n")

# Test Case 4: Lower bit-width (using int4)
def get_q_scale_and_zero_point_int4(tensor):
    q_min, q_max = -8, 7  # Range for 4-bit signed integer
    r_min, r_max = tensor.min().item(), tensor.max().item()
    scale = (r_max - r_min) / (q_max - q_min)
    zero_point = q_min - (r_min / scale)
    zero_point = int(round(zero_point))
    return scale, zero_point

def test_quantization_error_int4(tensor):
    scale, zero_point = get_q_scale_and_zero_point_int4(tensor)
    print(f"Scale: {scale}, Zero Point: {zero_point}")
    quantized_tensor = linear_q_with_scale_and_zero_point(tensor, scale, zero_point, dtype=torch.int32)  # Using int32 to store int4 values
    dequantized_tensor = linear_dequantization(quantized_tensor, scale, zero_point)
    print("Original Tensor:\n", tensor)
    print(f"Quantized Tensor:\n{quantized_tensor}")
    print(f"Dequantized Tensor:\n{dequantized_tensor}")
    error = (dequantized_tensor - tensor).square().mean()
    return error.item()

print("Test Case4 (int4):")
tensor4 = torch.tensor([[191.6, -13.5, 728.6],
                        [92.14, 295.5, -184],
                        [0, 684.6, 245.5]])
error4 = test_quantization_error_int4(tensor4)
print("\n")

print(f"Test Case 1 Error: {error1}")
print(f"Test Case 2 Error: {error2}")
print(f"Test Case 3 Error: {error3}")
print(f"Test Case 4 Error (int4): {error4}")
