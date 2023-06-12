'''
Author: dnimo kuochingcha@gmail.com
Date: 2023-06-11 15:47:22
LastEditors: dnimo kuochingcha@gmail.com
LastEditTime: 2023-06-11 21:29:47
FilePath: /ZhongYi-NER/test_Modules.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from Modules.Email import Email

mail = Email(receivers="zhangguoqingself@outlook.com")

mail.generator(text="这是一封测试邮件", subject="this is a test message")

mail.send()