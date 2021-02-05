import json

class Params:
    def __init__(self, json_path):
        self.path = json_path
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def __str__(self):
        # return string representation of 'Parameters' class
        # print(Parameters) or str(Parameters)
        ret = '======== [Config] ========\n'
        for k in self.__dict__:
            ret += '%s: %s\n' % (str(k), str(self.__dict__[k]))
        ret += '\n'
        return ret

    @property
    def dict(self):
        """
        Gives dict-like access to params instance by
        `params.dict['learning_rate']
        """
        return self.__dict__

    def update_dict(self, k, v):
        self.__dict__[k] = v