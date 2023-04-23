from lptorch.utils import is_tensorcore_int8_available, is_tensorcore_int8_available_offline
available_bits = [2, 3, 4, 8, '8:tc', '8:tc-li', 16]
def get_available_bits():
    cutlass_available = is_tensorcore_int8_available()
    if cutlass_available:
        return available_bits
    else:
        return [2, 3, 4, 8, 16]

def get_available_bits_offline(device_name):
    cutlass_available = is_tensorcore_int8_available_offline(device_name)
    if cutlass_available:
        return available_bits
    else:
        return [2, 3, 4, 8, 16]