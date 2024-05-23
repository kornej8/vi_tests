# from nipyapi import canvas,config
#
#
# config.nifi_config.host =  'https://10.101.16.31:8080/nifi-api'
# root_id = canvas.get_root_pg_id()
# root_process_group = canvas.get_process_group(root_id, 'id')
# new_processor_group = canvas.create_process_group(root_process_group, 'АНДРЕЙ ПРИВЕТ', (2000, 2000), 'this is a test')
# print(nipyapi.nifi.ProcessGroupsApi().get_process_groups(nipyapi.canvas.get_root_pg_id()))
import arviz as az
import matplotlib.pyplot as plt
import cloudpickle

pickle_filepath = f'F:/Python/vkr_shchepalov/my_018model10000.pkl'
with open(pickle_filepath , 'rb') as buff:
     mm = cloudpickle.load(buff)


tracker_res = mm['tracker_res']
result = mm['result']

fig = plt.figure(figsize=(16, 9))
mu_ax = fig.add_subplot(221)
std_ax = fig.add_subplot(222)
hist_ax = fig.add_subplot(212)
hist_ax.plot(result.hist)
hist_ax.set_title("Negative ELBO track");

plt.show()
