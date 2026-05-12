def format_bytes(num_bytes):
    if num_bytes == 0:
        return "0"
    return "{:,}".format(num_bytes)


def calculate_tensor_memory(tensor):
    return tensor.nelement() * tensor.element_size()