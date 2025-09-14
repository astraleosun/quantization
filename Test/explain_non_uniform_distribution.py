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

# Non-uniform distribution tensor
tensor = torch.tensor([[1.0000, 1.1000, 1.2000],
                       [1.3000, 1.4000, 1.5000],
                       [10.0000, 10.1000, 10.2000]])

scale, zero_point = get_q_scale_and_zero_point(tensor)
quantized_tensor = linear_q_with_scale_and_zero_point(tensor, scale, zero_point)
dequantized_tensor = linear_dequantization(quantized_tensor, scale, zero_point)

print(f"Scale: {scale}")
print(f"Zero Point: {zero_point}")
print(f"Original Tensor: {tensor}")
print(f"Quantized Tensor: {quantized_tensor}")
print(f"Dequantized Tensor: {dequantized_tensor}")

# Calculate error
error = (dequantized_tensor - tensor).square().mean()
print(f"Error: {error.item()}")

# Analyze error type
rounding_error = 0
clipping_error = 0
for i in range(tensor.shape[0]):
    for j in range(tensor.shape[1]):
        dequant_value = dequantized_tensor[i, j].item()
        orig_value = tensor[i, j].item()
        quant_value = quantized_tensor[i, j].item()
        dequantized_quant_value = linear_dequantization(torch.tensor([quant_value]), scale, zero_point).item()
        
        # Rounding error is the error due to rounding to the nearest quantized value
        rounding_error += (dequantized_quant_value - orig_value) ** 2
        # Clipping error is the error due to values being clipped to the min/max quantized values
        if quant_value == torch.iinfo(torch.int8).min or quant_value == torch.iinfo(torch.int8).max:
            clipping_error += (dequant_value - orig_value) ** 2

rounding_error /= tensor.numel()
clipping_error /= tensor.numel()

print(f"Rounding Error: {rounding_error}")
print(f"Clipping Error: {clipping_error}")
