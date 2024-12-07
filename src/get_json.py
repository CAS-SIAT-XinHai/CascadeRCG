import re

def change_to_json(res):
    pattern = r'\{([^}]*)\}'
    match = re.search(pattern, res, re.DOTALL)
    content = match.group(0)  # 提取匹配到的内容
    return content