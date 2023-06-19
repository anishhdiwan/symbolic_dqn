import configparser

config = configparser.ConfigParser()

config.read('GP_symbolic_DQN_config.ini')

for each_section in config.sections():
    for (each_key, each_val) in config.items(each_section):
        print(each_key)
        print(each_val)