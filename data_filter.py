# 首先我们第一步是把arch里的数据过滤出来，我们的数据名称的模式是{student_name_abbr}{number}_pHIT_lefteye.plist

import os
import re
import shutil

base_path = os.path.dirname(__file__)
arch_path = os.path.join(base_path, 'arch')
data_path = os.path.join(base_path, 'data')

if not os.path.exists(data_path):
    os.makedirs(data_path)
    
for file in os.listdir(arch_path):
    if file.endswith('.plist'):
        match = re.match(r'^[a-zA-Z]+[0-9]+[_-]+[0-9]+[_-]pHIT_(lefteye|righteye)\.plist$', file)
        if match:
            name = re.match(r'^([a-zA-Z]+)', file).group(1)
            # cp to data path
            name_path = os.path.join(data_path, name)
            if not os.path.exists(name_path):
                os.makedirs(name_path)
            shutil.copy(os.path.join(arch_path, file), os.path.join(name_path, file))

