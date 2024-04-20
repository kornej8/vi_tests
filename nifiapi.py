from nipyapi import canvas,config


config.nifi_config.host =  'https://10.101.16.31:8080/nifi-api'
root_id = canvas.get_root_pg_id()
root_process_group = canvas.get_process_group(root_id, 'id')
new_processor_group = canvas.create_process_group(root_process_group, 'АНДРЕЙ ПРИВЕТ', (2000, 2000), 'this is a test')
# print(nipyapi.nifi.ProcessGroupsApi().get_process_groups(nipyapi.canvas.get_root_pg_id()))