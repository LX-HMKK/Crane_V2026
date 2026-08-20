DEFAULT_BEAN_TO_DIGIT = {
    "yellow": 1,  # 黄豆
    "green": 2,   # 绿豆
    "white": 3,   # 白芸豆
}


def bean_name_to_digit(name, mapping=None):
    if mapping is None:
        mapping = DEFAULT_BEAN_TO_DIGIT
    return mapping.get(name)
