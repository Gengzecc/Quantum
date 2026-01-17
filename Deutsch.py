from cqlib import TianYanPlatform
from cqlib.circuits import Circuit
import numpy as np
from datetime import datetime
from cqlib.utils import LaboratoryUtils

'''指定量子计算机'''
login_key = ""
platform = TianYanPlatform(login_key=login_key)
computer_list_data = platform.query_quantum_computer_list()
for computer_data in computer_list_data:
    print(computer_data)
platform.set_machine("tianyan_sw")

'''编写量子线路'''
circuit = Circuit(qubits=[0, 1])

circuit.x(1)
circuit.h(0)
circuit.h(1)

# oracle 随机uf，小于0.5为constant，反之则为balanced
rnd = np.random.rand()
if rnd < 0.5:
    # two x gate equal i gate
    circuit.x(0)
    circuit.x(0)
    circuit.x(1)
    circuit.x(1)

else:
    # apperence h cz h
    circuit.cx(0, 1)

circuit.h(0)
circuit.measure(0)

print(circuit.qcis)
print(Circuit.load)

'''量子实验集创建'''
# create_lab 方法创建了一个实验集，并返回了该实验集的唯一标识 lab_id
lab_id = platform.create_lab(name=f'lab.{datetime.now().strftime("%Y%m%d%H%M%S")}', remark='test_collection')

print(lab_id)

'''实验运行'''
exp_id = platform.save_experiment(lab_id=lab_id, circuit=circuit.qcis, name=f'exp.{datetime.now().strftime("%Y%m%d%H%M%S")}')
query_id_single = platform.run_experiment(exp_id=exp_id, num_shots=5000)
print(f'query_id: {query_id_single}')

exp_result = platform.query_experiment(query_id=query_id_single, max_wait_time=120, sleep_time=5)
#返回值为list，包含若干字典形式，
    #key："resultStatus"为线路执行的原始数据，共计1+num_shots个数据，第一个数据为测量的比特编号和顺序，如本例中[0, 6]，其余为每shot对应的结果，每shot结果按照比特顺序排列。
    #key："probability"为线路测量结果的概率统计，经过实时的读取修正后的统计结果。
    #key："experimentTaskId"为本次实验的查询id，主要用于批量实验时的结果对应确认。
    #当测量比特大于15个时，结果统计对服务器要求较高，传递数据率也较大，故"probability"返回为空，请用户根据原始数据，配合当时量子计算机的读出保真度自行做修正。相关修正函数在高阶教程中有示例。用户也可以自己完善修正函数。

for res_name, res_data in exp_result[0].items():
    print(f"{res_name} : {res_data}")

'''实验结果统计'''
lu = LaboratoryUtils()
#将结果的全部空间进行统计
probability_whole=lu.readout_data_to_state_probabilities_whole(result=exp_result[0])
print(f'结果的全部空间统计: {probability_whole}')

#只对已有结果进行统计，概率为0的结果将不出现。
probability_part=lu.readout_data_to_state_probabilities_part(result=exp_result[0])
print(f'结果的部分空间统计: {probability_part}')