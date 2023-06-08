# -*- coding : UTF-8 -*-
# @file   : readYaml.py
# @Time   : 2023-06-08 19:37
# @Author : wmz

import yaml

with open('./yolov8.yaml', 'r', encoding='utf-8') as f:
    result = yaml.load(f.read(), Loader=yaml.FullLoader)
print(result)
print(type(result))
print(result['nc'], type(result['nc']))
print(result['scales'], type(result['scales']))

print(result['scales']['n'], type(result['scales']['n']))

print(result['backbone'], type(result['backbone']))
print(len(result['backbone']))

print(result['backbone'][0])
print(len(result['backbone'][0]))
print(result['backbone'][0][2], type(result['backbone'][0][2]))
print(result['backbone'][0][3], type(result['backbone'][0][3]))

