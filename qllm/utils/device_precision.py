from lptorch.utils import is_tensorcore_int8_available
available_bits = [2, 4, 8, '8:tc', '8:tc-li']
def get_available_bits():
    cutlass_available = is_tensorcore_int8_available()
    if cutlass_available:
        return available_bits
    else:
        return [2, 4, 8]