
import json

def data_offset(data):
    '''
    将数据整体偏移到0_1为0
    '''
    offset_data = json.loads(json.dumps(data))  # 深拷贝数据
    
    for user_name in offset_data:
        for exp_type in offset_data[user_name]:
            for trial_num in offset_data[user_name][exp_type]:
                offset = offset_data[user_name][exp_type][trial_num]['0_1']['mean']
                for platform_value in offset_data[user_name][exp_type][trial_num]:
                    for key in ['median', 'mean', 'q1', 'q3', 'min', 'max']:
                        offset_data[user_name][exp_type][trial_num][platform_value][key] -= offset
                        
                    offset_data[user_name][exp_type][trial_num][platform_value]['data'] = [x - offset for x in offset_data[user_name][exp_type][trial_num][platform_value]['data']]
                    
    return offset_data

