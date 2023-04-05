def convert_to_unit(data, unit):
    unit = unit.upper()
    if unit == 'MB':
        return data / 1024 / 1024
    elif unit == 'KB':
        return data / 1024
    elif unit == 'B':
        return data
    elif unit == 'GB':
        return data / 1024 / 1024 / 1024
    else:
        raise ValueError('unit should be one of MB, KB, B')