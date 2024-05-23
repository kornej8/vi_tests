params_kv = {}
params_part = False
skip_print = False
print_flag = False
show_next = False
val = None
replace_func = False
print_params = False
print_row = True


def return_uniq_func_name(lst, func_name, sep='', postfix=0):
    check_func_name = f"{func_name}{sep}{'' if postfix == 0 else postfix}"
    return return_uniq_func_name(lst, func_name, '_', postfix+1) if check_func_name in lst else check_func_name


def create_func_name(params_list_old):
    lst = params_list_old
    fncs = []
    for i, l in enumerate(lst):
        if lst[i][0].startswith('$'):
            continue
        lst[i] = list(filter(lambda x: False if x == '' else True, list(map(lambda x: x.strip(), l))))
        phs = lst[i][:-1][0].split(',')
        l_num = phs[-1].split(';')[1][0] if ';' in phs[-1] else 0
        func_name = f"{phs[0].split('(')[1]}_{phs[0].split('(')[0]}_L{l_num}"

        uniq_func_name = return_uniq_func_name(fncs, func_name)
        fncs.append(uniq_func_name)
    return fncs


def create_params_name(params_list_old):
    lst = params_list_old
    fncs = []
    for i, l in enumerate(lst):
        if lst[i][0].startswith('$'):
            continue
        lst[i] = list(filter(lambda x: False if x == '' else True, list(map(lambda x: x.strip(), l))))
        phs = lst[i][:-1][0].split(',')
        l_num = phs[-1].split(';')[1][0] if ';' in phs[-1] else 0
        func_name = f"{phs[0].split('(')[1]}_{phs[0].split('(')[0]}_L{l_num}"
        uniq_func_name = return_uniq_func_name(fncs, func_name)
        fncs.append(uniq_func_name)
    return fncs

with open('test_data/CoCr-01Oik.tdb', 'r+') as tdb:
    with open('test_data/CoCr-01Oik_with_funcs.tdb', 'w') as file:

        params = []
        lines = enumerate(tdb.readlines())
        if True:
            for i, line in lines:
                if show_next:
                    val += line.strip()
                    params.append(val.replace('PAR', '').strip().replace('\n', '').split(',,'))
                    print_row = False
                    show_next = False
                    val = None
                else:
                    if line.strip().startswith('PAR') or line.strip().startswith('$PAR'):
                        params_part = True
                        if line.strip()[-1] == '!':
                            show_next = False
                            params.append(line.replace('PAR', '').strip().replace('\n', '').split(',,'))
                            print_row = False
                        else:
                            val = line
                            show_next = True
                    else:
                        if line.strip() == '$' and params_part:
                            params_part = False
                        else:
                            if not params_part:
                                print_flag = True
                        if print_flag and not params_part:
                            params_list = create_func_name(params)
                            for param in params_list:
                                print(f" FUNCTION {param}  6000 N !", file=file, end = '\n')
                            params = []
                            print_flag = False
                    if print_row:
                        print(line, file=file, end = '')
