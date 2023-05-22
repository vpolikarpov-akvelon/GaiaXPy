
def has_continuous_key(result):
    return bool([key for key in result.keys() if 'continuous' in key.lower()])