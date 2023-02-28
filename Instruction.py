import logging
import time
import sys

class Instruction:
    """Controller 具备的指令"""
    logger = None
    MoveIns=0#指令类型常量，0=控制无人机移动的指令
    StateIns=1#1=不会移动无人机，仅修改无人机的状态
    @staticmethod
    def setLogger(input_logger=None):
        if input_logger is None:
            logger = logging.getLogger("Test loger")
            logger.setLevel(logging.INFO)  # 设置默认的级别
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(logging.Formatter(fmt='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
                                              datefmt="%H:%M:%S"))
            logger.addHandler(ch)
            Instruction.logger = logger
        else:
            Instruction.logger = input_logger

    def __init__(self, name, action_func, instruction_type, kwargs={}):
        """设置指令名、指令执行函数、所需的参数"""
        self.name = name
        self.__action_func = action_func
        self.ins_type=instruction_type#
        self.__action_func_kwargs = kwargs

    def __repr__(self):
        return f"instruction : name={self.name},act={self.__action_func.__name__}"


    def act(self):
        """指令执行，调用指令绑定的函数"""
        Instruction.logger.info(f"Instruction:{self.name} Acting!")
        self.__action_func(**self.__action_func_kwargs)  # 双星号（**）将参数以字典的形式导入:

def testfunc(time, year=2090):
    print(f"Test Instruction {year},{time}")

if __name__ == '__main__':
    Instruction.setLogger()
    ins1=Instruction("PrintTimeOnScreen", testfunc, {'year':2011, 'time':time.time()})
    ins1.act()
