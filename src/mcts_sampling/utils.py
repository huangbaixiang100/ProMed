import re
import json
from collections import defaultdict

def match_choice(text,options_dict):
    option = ["A", "B", "C", "D", "E", "F", "G"]
    pattern = r"(?:正确答案|答案|正确选项|answer)[：:\s]*(?:is|是|为|应该是|应该为)?\s*[【]?\s*([A-Ga-g]{1,7})\s*[】]?"

    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        answer = matches[0].upper()
        answer = "".join(sorted(set(answer)))
        return answer
    else:
        tmp=[]
        for op_letter, op_text in options_dict.items():
            if op_text in text:
                tmp.append(op_letter)
        return "".join(tmp)
    return "".join([i for i in text if i in option])