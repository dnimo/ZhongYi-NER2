count = 0

for count,line in enumerate(open(r"complete.json",'r',encoding='utf-8')):
    pass
    count += 1

print("文件总行数：",count)
split = 10 # 拆成五个文件
nums = [ (count*i//split) for i in range(1,split+1)]
print(nums)
# 拆分文件
import json
current_lines = 0
data_list = []
# 打开大文件，拆成小文件
with open('complete.json', 'r', encoding='utf-8') as file:
    i = 0
    for line in file:
        # line = line.replace('},','}')
        data_list.append(line)
        current_lines +=1
        if current_lines in nums:
            print(current_lines)
            # 保存文件
            file_name = 'data_' + str(current_lines) + '.json'
            with open(file_name,'w',encoding = 'utf-8') as f:
                #print(len(data_list))
                # data = json.dumps(data_list)
                # f.write(data)
                for l in data_list:
                    f.write(l)
                data_list = []
                data = []