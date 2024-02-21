
#-*-coding:gb2312-*-

from parser_util_v2.data_flow import *
from parser_util_v2.data_flow2 import *
from parser_util_v2.get_relation_set import *
from parser_util_v2.project_utils import *

import numpy as np
import pandas as pd
#encoding=utf-8

class Feature(object):
    """
    记录一个用户的输入指令：

    哪个记录
    第几个token
    length
    """
    def __init__(self, name, Opcode, Opcode_Len, Format_Len, self_cal = False, frags = [],  Armstr = "", Namespace = "", Size = 4, Set = "", Format = "", \
                 lets = {}, FM = ""):
        self.name = name
        self.Opcode = Opcode
        self.Opcode_Len = Opcode_Len
        self.Format_Len = Format_Len
        self.self_cal = self_cal
        self.frags = frags
        self.Armstr = Armstr
        self.Namespace = Namespace
        self.Size = Size
        self.Set = Set
        self.Format = Format
        self.lets = lets
        self.FM = FM

    def print_info(self):
        print("----------Feature Info-------------")
        print("Name: ", end = "")
        print(self.name)
        print("Armstr: ", end="")
        print(self.Armstr)
        print("Set: ", end="")
        print(self.Set)
        print("Format: ", end="")
        print(self.Format)
        print("Lets: ", end="")
        print(self.lets)
        print("----------Feature Info-------------")

class Frag_Split(object):
    """
    记录一个用户的输入指令的字段：
    """

    def __init__(self, id, frag_start_bit = -1, frag_end_bit = -1, ins_start_bit = -1, ins_end_bit = -1):
        self.id = id
        self.frag_start_bit = frag_start_bit
        self.frag_end_bit = frag_end_bit
        self.ins_start_bit = ins_start_bit
        self.ins_end_bit = ins_end_bit

    def print_info(self):
        print("----------Frag Split Info-------------")
        print("ID: ", end = "")
        print(self.id)
        print("Frag Start Bit: ", end="")
        print(self.frag_start_bit)
        print("Frag End Bit: ", end="")
        print(self.frag_end_bit)
        print("Instruction Start Bit: ", end="")
        print(self.ins_start_bit)
        print("Instruction End Bit: ", end="")
        print(self.ins_end_bit)
        print("----------Frag Split Info-------------")

class Register_Op(object):
    def __init__(self, Type = "Reg", Mode = "INT", ILLEGAL = []):
        self.Type = Type
        self.Mode = Mode
        self.ILLEGAL = ILLEGAL

    def print_info(self):
        print("----------Reg Info-------------")
        print("Type: ", end = "")
        print(self.Type)
        print("Mode: ", end="")
        print(self.Mode)
        print("----------Reg Info-------------")

class Immediate_Op(object):
    def __init__(self, Type = "Imm", is_signed = True, is_special = False, LSB = 0, NoZero = False):
        self.Type = Type
        self.is_signed = is_signed
        self.is_special = is_special
        self.LSB = LSB
        self.NoZero = NoZero

    def print_info(self):
        print("----------Immediate Info-------------")
        print("Type: ", end = "")
        print(self.Type)
        print("LSB: ", end="")
        print(self.LSB)
        print("----------Immediate Info-------------")

class Frag(object):
    """
    记录一个用户的输入指令的字段：
    """

    def __init__(self, name, start_bit = -1, end_bit = -1, is_split = False, frags_split = [], is_op = False, Value = None, Op = None):
        self.name = name
        self.start_bit = start_bit
        self.end_bit = end_bit
        self.is_split = is_split
        self.frags_split = frags_split
        self.is_op = is_op
        self.Value= Value
        self.Op = Op

    def print_info(self):
        print("----------Frag Info-------------")
        print("Name: ", end = "")
        print(self.name)
        print("Value: ", end="")
        print(self.Value)
        print("Start Bit: ", end="")
        print(self.start_bit)
        print("End Bit: ", end="")
        print(self.end_bit)
        print("----------Frag Info-------------")




def value_mapping(right_value):
    #right_value = attrs.split("=")[1]
    if right_value == "False" or right_value == "No":
        return False
    elif right_value == "True":
        return True
    elif right_value == "NULL" or right_value == "?":
        return None
    return right_value

def Reg_Construction(attr_list):
    flag = False
    Mode = "INT"
    ILLEGAL = []
    for attrs in attr_list:
        if attrs.split("=")[0] == "op_type" and attrs.split("=")[1] == "Reg":
            flag = True
        if attrs.split("=")[0] == "Mode":
            Mode = value_mapping(attrs.split("=")[1])
        if attrs.split("=")[0] == "ILLEGAL":
            ILLEGAL.clear()
            ILLEGAL.append(value_mapping(attrs.split("=")[1]))
    if flag:
        return Register_Op("Reg",Mode,ILLEGAL)
    return None

def Imm_Construction(attr_list):
    flag = False
    is_signed = True
    is_special = False
    LSB = 0
    NoZero = False
    for attrs in attr_list:
        if attrs.split("=")[0] == "op_type" and attrs.split("=")[1] == "Imm":
            flag = True
        if attrs.split("=")[0] == "is_signed":
            is_signed = value_mapping(attrs.split("=")[1])
        if attrs.split("=")[0] == "is_special":
            is_special = value_mapping(attrs.split("=")[1])
        if attrs.split("=")[0] == "LSB":
            LSB = value_mapping(attrs.split("=")[1])
        if attrs.split("=")[0] == "NoZero":
            NoZero = value_mapping(attrs.split("=")[1])
    if flag:
        return Immediate_Op("Imm",is_signed,is_special,LSB,NoZero)
    return None

def Imm_Construction(attr_list):
    flag = False
    is_signed = True
    is_special = False
    LSB = 0
    NoZero = False
    for attrs in attr_list:
        if attrs.split("=")[0] == "op_type" and attrs.split("=")[1] == "Imm":
            flag = True
        if attrs.split("=")[0] == "is_signed":
            is_signed = value_mapping(attrs.split("=")[1])
        if attrs.split("=")[0] == "is_special":
            is_special = value_mapping(attrs.split("=")[1])
        if attrs.split("=")[0] == "LSB":
            LSB = value_mapping(attrs.split("=")[1])
        if attrs.split("=")[0] == "NoZero":
            NoZero = value_mapping(attrs.split("=")[1])
    if flag:
        return Immediate_Op("Imm",is_signed,is_special,LSB,NoZero)
    return None

def IS_Similiar_Ins(Feature1, Feature2):
    if len(Feature1.frags) != len(Feature2.frags):
        return False
    for i in range(0, min(len(Feature1.frags),len(Feature2.frags))):
        if IS_Equ_Frags(Feature1.frags[i], Feature2.frags[i]) == -1:
         return False
    return True

def IS_Equ_Frags(frag1,frag2):
    if frag1.name != frag2.name:
        return -1
    if frag1.start_bit != frag2.start_bit:
        return -1
    if frag1.end_bit != frag2.end_bit:
        return -1
    if frag1.is_split != frag2.is_split:
        return -1
    if frag1.is_op != frag2.is_op:
        return -1
    if frag1.Op and frag2.Op:
        if frag1.Op.Type == "Reg" and frag2.Op.Type == "Imm":
            return -1
        elif frag1.Op.Type == "Imm" and frag2.Op.Type == "Reg":
            return -1
        elif frag1.Op.Type == "Reg" and frag2.Op.Type == "Reg":
            if frag1.Value != frag2.Value:
                return -1
        elif frag1.Op.Type == "Imm" and frag2.Op.Type == "Imm":
            if frag1.Value != frag2.Value:
                return -1
    if frag1.Value != frag2.Value:
        return 0
    return 1


def Diff_Frags(Feature1, Feature2):
    diff_num = []
    options = ["rd","rs1","rs2","offset","immediate"]
    need_copy = False
    if len(Feature1.frags) != len(Feature2.frags):
        len1 = max(len(Feature1.frags),len(Feature2.frags)) - 1
        len2 = min(len(Feature1.frags),len(Feature2.frags)) - 1
        while len1 != len2:
            diff_num.append(len1)
            len1 -= 1
    for i in range(0, min(len(Feature1.frags),len(Feature2.frags))):
        if IS_Equ_Frags(Feature1.frags[i], Feature2.frags[i]) == 0 and Feature1.frags[i].name not in options:
            diff_num.append(i)
        elif IS_Equ_Frags(Feature1.frags[i], Feature2.frags[i]) == 1 and Feature1.frags[i].name not in options and not Feature1.frags[i].Value:
            diff_num.append(i)
        elif IS_Equ_Frags(Feature1.frags[i], Feature2.frags[i]) == -1:
            need_copy = True
    return diff_num, need_copy




def small_frag_construction(attr_list):
    flag = False
    small_frag_list = []
    for attrs in attr_list:
        if attrs.split("=")[0] == "is_split" and attrs.split("=")[1] == "True":
            flag = True
        if attrs[0] == "{":
            attrs = attrs.replace("{","")
            attrs = attrs.replace("}", "")
            small_attr_list = attrs.split("#")
            id = -1
            frag_start_bit = -1
            frag_end_bit = -1
            ins_start_bit = -1
            ins_end_bit = -1
            for small_attrs in small_attr_list:
                if small_attrs.split("=")[0] == "frag_id":
                    id = int(value_mapping(small_attrs.split("=")[1]))
                if small_attrs.split("=")[0] == "frag_range":
                    frag_start_bit = int(small_attrs.split("=")[1].split("-")[0])
                    frag_end_bit = int(small_attrs.split("=")[1].split("-")[1])
                if small_attrs.split("=")[0] == "bit_range":
                    ins_start_bit = int(small_attrs.split("=")[1].split("-")[0])
                    ins_end_bit = int(small_attrs.split("=")[1].split("-")[1])
            small_frag_list.append(Frag_Split(id,frag_start_bit,frag_end_bit,ins_start_bit,ins_end_bit))
    if flag:
        #for s in small_frag_list:
            #s.print_info()
        return small_frag_list
    return None

def frag_split(input_str):
    """
    name, start_bit = -1, end_bit = -1, is_split = False, frags_split = [], is_op = False, Value = None, Op = None
    """
    input_str = input_str[1:]
    input_str = input_str[:-1]
    attr_list = input_str.split(";")
    name = ""
    start_bit = -1
    end_bit = -1
    is_split = False
    frags_split = []
    is_op = False
    Value = None
    Op = None
    for attrs in attr_list:
        if attrs[0] == "{":
            continue
        else:
            if attrs.split("=")[0] == "is_split":
                is_split = value_mapping(attrs.split("=")[1])
            if attrs.split("=")[0] == "name":
                name = value_mapping(attrs.split("=")[1])
            if attrs.split("=")[0] == "is_op":
                is_op = value_mapping(attrs.split("=")[1])
            if attrs.split("=")[0] == "value":
                Value = value_mapping(attrs.split("=")[1])
            if attrs.split("=")[0] == "bit_range" and attrs.split("=")[1] != "NULL":
                start_bit = int(value_mapping(attrs.split("=")[1].split("-")[0]))
                end_bit = int(value_mapping(attrs.split("=")[1].split("-")[1]))
    if is_op:
        Reg = Reg_Construction(attr_list)
        Imm = Imm_Construction(attr_list)
        if Reg != None:
            Op = Reg
            #Reg.print_info()
        elif Imm != None:
            Op = Imm
            #Imm.print_info()
    Small_Frag_Construction = small_frag_construction(attr_list)
    F = Frag(name,start_bit,end_bit,is_split,Small_Frag_Construction,is_op,Value,Op)
    #F.print_info()
    return F

def Feature_Construction():
    """
    name, Opcode, self_cal = False, frags = [],  Armstr = "", Namespace = "", Size = 4, Set = "", Format = "", let ={}
    """
    df = pd.read_csv('../Input/feature.csv')
    Feature_list = []
    for rows in df.itertuples():
        Name = getattr(rows, 'RName')
        Opcode = int(getattr(rows, 'Opcode'))
        Opcode_Len = int(getattr(rows, 'Opcode_Len'))
        Format_Len = int(getattr(rows,"Format_Len"))
        Self_Cal = value_mapping(getattr(rows, "Self_Cal"))
        frags = []
        for i in range(1,8):
            if getattr(rows, 'frag' + str(i)) != "None":
                frags.append(frag_split(getattr(rows, 'frag' + str(i))))
        Armstr = getattr(rows,"asmstr")
        NameSpace = getattr(rows,"Namespace")
        Size = int(getattr(rows,"Size"))
        FM = getattr(rows,"FM_Father")
        Format = getattr(rows,"Format")
        Set = getattr(rows,"Set")
        lets = {}
        for let_info in ["hasSideEffects","mayLoad","mayStore","isCall","hasDelaySlot","isRematerializable","isAsCheapAsMove",\
                         "isBarrier","isReturn","isTerminator","isBranch","SoftFail","isComm","Arch","isCTI","hasForbiddenSlot",\
                         "IsPCRelativeLoad","hasFCCRegOperand","DecoderNamespace","TwoOperandAliasConstraint","TSFlags_FormBits",\
                         "TSFlags_isCTI","TSFlags_hasForbiddenSlot","TSFlags_IsPCRelativeLoad","TSFlags_hasFCCRegOperand"]:
            lets[let_info] = value_mapping(getattr(rows,let_info))

        F = Feature(Name,Opcode,Opcode_Len, Format_Len, Self_Cal,frags,Armstr,NameSpace,Size,Set,Format,lets,FM)
        #F.print_info()
        Feature_list.append(F)
    return Feature_list



def Find_Instruction(ins_name, Feature_List):
    for Features in Feature_List:
        if Features.name == ins_name:
            return Features
    return None


if __name__ == "__main__":
    Feature_Construction()
    #frag_split("{is_split=True;name=offset;value=NULL;{frag_id=1#frag_range=12-12#bit_range=31-31};{frag_id=2#frag_range=10-5#bit_range=30-25};{frag_id=3#frag_range=4-1#bit_range=11-8};{frag_id=4#frag_range=5-5#bit_range=7-7};is_op=True;op_type=Imm;is_signed=True;is_special=False;LSB=1;NoZero=False}")
