class CreateTDBapi:
    """

    Класс создает уникальные функции из определений параметров ТДБ файла
    В результате создается новый ТДБ файл к которому будет обращаться алгоритм.
    Новый файл имеет название как у исходного ТДБ файла с постфиксом *_modified.tdb

    > 'CoCr-01Oik.tdb' # Исходный ТДБ файл
    > CreateTDBapi('CoCr-01Oik.tdb') #Создание API внутри ТДБ файла для обащений через PyCalphad
    > 'CoCr-01Oik_modified.tdb'  # ТДБ файл с функциями вместо параметров

    """
    def __init__(self, filename):
        pars_counter = 0
        with open(filename, 'r+') as tdb:
            if filename[-4:].lower() != '.tdb':
                raise Exception('Файл должен быть в формате ".tdb"')
            else:
                new_filename = filename[:-4] + '_modified.tdb'
            with open(new_filename, 'w') as file:

                params, funcs_names, functions = [], [], []
                lines = enumerate(tdb.readlines())
                if True:
                    for i, line in lines:
                        if line.strip().startswith('PAR'):
                            pars_counter += 1
                            if line.strip()[-1] == '!':
                                row = line.replace('PAR', '').strip().replace('\n', '').split(',,')

                                func_name = CreateTDBapi.generate_line_name(row, pars_counter)
                                if func_name is not None:
                                    row_func = line.split(',,')
                                    par_value = CreateTDBapi.generate_values(row)
                                    funcs_names.append(func_name)
                                    params.append(row)
                                    row_param = par_value if par_value.startswith('-') or par_value.startswith('+') else '+' + par_value

                                    rows, funcs = CreateTDBapi.create_sub_params(row_param, func_name)
                                    row_func[1] = f"                {rows};"

                                    for func in funcs:
                                        function_def = f" FUNCTION {func.split('|')[0]}{' '* 10}{func.split('|')[1]};{' '*25}6000 N !"

                                        functions.append(function_def)
                                    print(',,'.join(row_func), file=file, end='\n')
                                else:
                                    print(line, file=file, end='')

                            else:
                                print(line, file=file, end='')
                                continue

                        else:
                            if line.startswith('$ ---------------'):
                                for i in functions:
                                    print(i, file=file, end='\n')
                                print(line, file=file, end='')
                                params, funcs_names, functions = [], [], []
                            else:
                                print(line, file=file, end='')

    @staticmethod
    def create_sub_params(row, param_name):
        funcs, sub_param_list, sub_param_names, params_counter = [], [], [], 0

        num = ''
        for j, i in enumerate(row):

            if i.isdigit() or i == '.':
                num += i

            if i in ['-', '+', '*']:
                if num != '':
                    sub_param_list.append(num)
                    params_counter += 1
                    sub_param_names.append(f"{param_name}_V{params_counter}")

                num = i
        if num != '' and num != '+' and num != '-' and num != '*' and num != 'T':
            sub_param_list.append(num)
            params_counter += 1
            sub_param_names.append(f"{param_name}_V{params_counter}")

        for i, value in enumerate(sub_param_list):
            row = row.replace(value, '+' + sub_param_names[i])
            funcs.append(f"{sub_param_names[i]}|{'+' + value if value[0] != '-' and value[0] != '+' else value}")
        return row, funcs

    @staticmethod
    def generate_line_name(line, pars_counter):
        line_clear = list(map(lambda x: x.strip(), line))
        value = line_clear[1].strip().replace(';', '')
        try:
            eval(value.replace('T', '1'))
            func_name = line_clear[0]
            func = f"{func_name.split('(')[0]}_{func_name.split('(')[1].split(',')[0]}_{pars_counter}"
            if ';' in func_name:
                l_num = func_name.split(';')[-1].replace(')', '')
                return f"{func}_L{l_num}"
            else:
                return func
        except:
            return None

    @staticmethod
    def generate_values(line):
        line_clear = list(map(lambda x: x.strip(), line))
        value = line_clear[1].strip().replace(';', '')
        try:
            eval(value.replace('T', '1'))
            return value
        except:
            return ''
