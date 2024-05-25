import functools
from os import listdir

class AcceptChoice:

    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func

    def __call__(self, *args, **kwargs):
        var = self.func(*args, **kwargs)
        ans = input(f"\nВы внесли {var}. Подтвердите выбор (введите Y/N)   ").lower()
        if ans == 'y':
            return var
        else:
            return __class__(self.func)(*args, **kwargs)


class ConfigurateApp:
    """

    Класс для инициализации некоторых параметров для работы алгоритма

    """
    def __init__(self):

        print('\nПривет! \nДля сопоставления ТДБ файлов далее необходимо указать некоторую информацию ниже./')

        get = lambda var: ConfigurateApp.get_from_input(var)
        self.el = get('Введите исследуемый элемент:   ').upper()
        raw_path = get(
            '\nУкажите путь до директории с термодинамическими базами данных:\nВ папке должны находиться только ТДБ файлы!\n (Пример: C:/main_folder/tdb_folder/)         :')
        path = ConfigurateApp.get_choised_path(raw_path)

        flag = True
        while flag:
            try:
                tdb_names = listdir(path)
                flag = False
                self.tdb_path = path
            except:
                print('\nУпс! Такого пути не существует!')
                raw_path = get(
                    'Укажите новый путь до директории с термодинамическими базами данных:\nВ папке должны находиться ТДБ файлы!\n (Пример: C:/main_folder/tdb_folder/)         :')
                path = ConfigurateApp.get_choised_path(raw_path)

        tdbs = ', '.join([f'({i + 1}) {name}' for i, name in enumerate(tdb_names)])
        if len(tdb_names)>=2:
            first_tdb_num = get(f'Укажите номер первого ТДБ файла для сопоставления \n{tdbs}:   ')
        else:
            print('\nНе найдено достаточное количество ТДБ файлов!')

        if len(tdb_names)>=2:
            second_tdb_num = get(f'Укажите номер второго ТДБ файла для сопоставления \n{tdbs}:   ')
        else:
            print('\nНе найдено достаточное количество ТДБ файлов!')
        try:
            self.tdb_first_name = tdb_names[int(first_tdb_num)-1]
            self.tdb_second_name = tdb_names[int(second_tdb_num)-1]
            print(
                f"\nВы выбрали {tdb_names[int(first_tdb_num) - 1]} и {tdb_names[int(second_tdb_num) - 1]} для сопоставления.")
        except:
            print('Введите корректные номер ТДБ файлов! Перезапустите скрипит и введите все заново.')

        raw_path = get(
            'Укажите путь до директории с экспериментальными данными:\nОпытные данные должны быть в формате Excel\n (Пример: C:/main_folder/experiment_folder/)         :')
        path = ConfigurateApp.get_choised_path(raw_path)

        flag = True
        while flag:
            try:
                file_names = listdir(path)
                self.data_path = path
                flag = False
            except:
                print('\nУпс! Такого пути не существует!')
                raw_path = get(
                    'Укажите новый путь до директории с экспериментальными данными:\nОпытные данные должны быть в формате Excel\n (Пример: C:/main_folder/experiment_folder/)         :')
                path = ConfigurateApp.get_choised_path(raw_path)

        data = ', '.join([f'({i + 1}) {name}' for i, name in enumerate(file_names)])
        if len(file_names) >= 1:
            data_num = get(f'Укажите номер файла c экспериментальными данными \n{data}:   ')
        else:
            print('\nНе найдено достаточное количество файлов!')

        try:
            self.experemental_data = file_names[int(data_num) - 1]
            print(
                f"\nВы выбрали файл {file_names[int(data_num) - 1]} в качестве эксперементальных данных.\n")
            print('Все необходимые данные внесены! Спасибо. Можно запускать расчет показателей алгоритма.')
        except:
            print('Введите корректный номер файла! Перезапустите скрипт и введите все заново.')

    @staticmethod
    @AcceptChoice
    def get_from_input(text):
        return input(text)

    @staticmethod
    def get_choised_path(path):
        path = path.replace(f'\\', '/')
        if path[-1] != '/':
            path = path + '/'
        return path

